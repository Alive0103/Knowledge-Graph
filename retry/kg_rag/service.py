from __future__ import annotations

import json
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from alignment.dbp15k import DBP15KDataset, EntityRecord, KgKey, RelationRecord
from alignment.evaluation import evaluate_final_model_alignment, evaluate_raw_alignment
from entity_linking.es_eval import text_search
from entity_linking.es_index import create_es_client

from .config import KgRagConfig


ENTITY_KEY_RE = re.compile(r"^(?:kg)?(?P<kg>[12])[:#](?P<entity_id>\d+)$", re.IGNORECASE)
RELATION_KEY_RE = re.compile(
    r"^(?:rel|relation|关系)[:\s_-]*(?P<kg>[12])[:#](?P<relation_id>\d+)$",
    re.IGNORECASE,
)


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _normalize_text(text: object) -> str:
    return " ".join(str(text or "").strip().split())


def _casefold_text(text: object) -> str:
    return _normalize_text(text).replace("_", " ").casefold()


def _score_text_match(query: str, value: str) -> int | None:
    needle = _casefold_text(query)
    haystack = _casefold_text(value)
    if not needle or not haystack:
        return None
    if needle == haystack:
        return 0
    if haystack.startswith(needle):
        return 1
    if needle in haystack:
        return 2
    return None


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _split_dataset_name(dataset_name: str) -> tuple[str, str]:
    parts = (dataset_name or "").split("_", maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "kg1", "kg2"


def _kg_to_side(dataset_name: str, kg: KgKey) -> str:
    left, right = _split_dataset_name(dataset_name)
    return left if kg == "1" else right


def _side_to_kgs(dataset_name: str, kg_side: str) -> list[KgKey]:
    normalized = (kg_side or "auto").strip().lower()
    left, right = _split_dataset_name(dataset_name)
    if normalized in {"1", left.lower()}:
        return ["1"]
    if normalized in {"2", right.lower()}:
        return ["2"]
    if normalized == "zh":
        return ["1"] if left.lower() == "zh" else ["2"]
    if normalized == "en":
        return ["1"] if left.lower() == "en" else ["2"]
    return ["1", "2"]


def _entity_key(kg: KgKey, entity_id: int) -> str:
    return f"{kg}:{int(entity_id)}"


def _triple_key(triple: dict[str, object]) -> tuple[object, ...]:
    return (
        triple.get("kg"),
        triple.get("head_id"),
        triple.get("relation_id"),
        triple.get("tail_id"),
    )


class KgRagService:
    def __init__(self, config: KgRagConfig | dict[str, Any] | None = None) -> None:
        if isinstance(config, KgRagConfig):
            self.config = config
        else:
            self.config = KgRagConfig.from_dict(config)
        self._dataset: DBP15KDataset | None = None
        self._es = None

    @property
    def dataset(self) -> DBP15KDataset:
        if self._dataset is None:
            self._dataset = DBP15KDataset(
                dataset_dir=self.config.dataset_dir,
                dataset_name=self.config.dbp15k_dataset,
            )
        return self._dataset

    @property
    def dataset_name(self) -> str:
        return self.dataset.dataset_name

    def healthcheck(self) -> dict[str, Any]:
        required_paths = self.config.validate_required_paths()
        es_ok = False
        es_error = ""
        try:
            self._get_es_client()
            es_ok = True
        except Exception as exc:  # noqa: BLE001
            es_error = str(exc)
        return {
            "ok": not required_paths,
            "timestamp": _utc_now(),
            "required_path_issues": required_paths,
            "es_available": es_ok,
            "es_error": es_error,
            "dataset": self.config.dbp15k_dataset,
        }

    def _get_es_client(self):
        if self._es is not None:
            return self._es
        client = create_es_client(es_url=self.config.es_url)
        client.info()
        self._es = client
        return self._es

    def _parse_entity_key(self, value: str) -> tuple[KgKey, int] | None:
        match = ENTITY_KEY_RE.match(_normalize_text(value))
        if not match:
            return None
        return match.group("kg"), int(match.group("entity_id"))

    def _parse_relation_key(self, value: str) -> tuple[KgKey, int] | None:
        match = RELATION_KEY_RE.match(_normalize_text(value))
        if not match:
            return None
        return match.group("kg"), int(match.group("relation_id"))

    def _search_entities_local(self, query: str, kg_side: str = "auto", limit: int = 10) -> list[tuple[EntityRecord, float]]:
        candidates: list[tuple[int, float, EntityRecord]] = []
        for kg in _side_to_kgs(self.dataset_name, kg_side):
            for entity in self.dataset.entities[kg].values():
                score = _score_text_match(query, entity.name)
                if score is None:
                    continue
                candidates.append((score, float(1.0 / (1 + score)), entity))
        candidates.sort(key=lambda item: (item[0], item[2].kg, item[2].entity_id))
        return [(entity, score) for _, score, entity in candidates[:limit]]

    def _search_relations_local(
        self,
        query: str,
        kg_side: str = "auto",
        limit: int = 10,
    ) -> list[tuple[RelationRecord, float]]:
        results: list[tuple[int, float, RelationRecord]] = []
        for kg in _side_to_kgs(self.dataset_name, kg_side):
            for relation in self.dataset.search_relations(kg, query, limit=limit):
                score = _score_text_match(query, relation.name)
                if score is None and _normalize_text(query).isdigit():
                    score = 0
                if score is None:
                    score = 2
                results.append((score, float(1.0 / (1 + score)), relation))
        results.sort(key=lambda item: (item[0], item[2].kg, item[2].relation_id))
        return [(relation, score) for _, score, relation in results[:limit]]

    def _search_entity_linking(self, query: str, limit: int = 10) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        processing = {"es_available": False, "es_error": "", "fallback_used": False}
        try:
            es = self._get_es_client()
            hits = text_search(es=es, index_name=self.config.es_index_name, query=query, limit=limit)
            processing["es_available"] = True
            return hits, processing
        except Exception as exc:  # noqa: BLE001
            processing["fallback_used"] = True
            processing["es_error"] = str(exc)
            return [], processing

    def _format_entity(
        self,
        entity: EntityRecord,
        match_score: float = 1.0,
        match_source: str = "dbp15k_name",
    ) -> dict[str, Any]:
        side = _kg_to_side(self.dataset_name, entity.kg)
        key = _entity_key(entity.kg, entity.entity_id)
        return {
            "entity_name": entity.name,
            "entity_type": "dbp15k_entity",
            "description": f"{side} entity {entity.entity_id}",
            "entity_id": entity.entity_id,
            "kg": entity.kg,
            "kg_side": side,
            "match_score": round(match_score, 4),
            "match_source": match_source,
            "detail_route": key,
            "source_id": key,
        }

    def _format_relation(
        self,
        relation: RelationRecord,
        match_score: float = 1.0,
        sample_triple_count: int = 0,
    ) -> dict[str, Any]:
        side = _kg_to_side(self.dataset_name, relation.kg)
        return {
            "relation_id": relation.relation_id,
            "relation_name": relation.name,
            "match_score": round(match_score, 4),
            "sample_triple_count": sample_triple_count,
            "description": f"{side} relation {relation.relation_id}",
            "kg": relation.kg,
            "kg_side": side,
            "source_id": f"rel:{relation.kg}:{relation.relation_id}",
        }

    def _format_alignment(
        self,
        source: EntityRecord,
        split: str,
        target: EntityRecord,
        score: float,
        evidence: str,
    ) -> dict[str, Any]:
        return {
            "source_entity": self._format_entity(source, match_source="alignment_source"),
            "target_entity": self._format_entity(target, match_source="alignment_target"),
            "split": split,
            "score": round(score, 4),
            "evidence": evidence,
        }

    def _entity_summary_chunks(
        self,
        detail: dict[str, Any],
        relation_limit: int,
        triple_limit: int,
    ) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        relation_items = detail.get("relation_summary", [])[:relation_limit]
        if relation_items:
            chunks.append(
                {
                    "chunk_id": f"entity:{detail['kg']}:{detail['entity_id']}:relations",
                    "title": f"Entity {detail['name']} relation summary",
                    "content": "; ".join(
                        f"{item['relation_name']}({item['relation_id']}): {item['count']}" for item in relation_items
                    ),
                    "source": "dbp15k",
                    "source_id": f"{detail['kg']}:{detail['entity_id']}",
                }
            )
        triple_items = detail.get("triples", [])[:triple_limit]
        if triple_items:
            chunks.append(
                {
                    "chunk_id": f"entity:{detail['kg']}:{detail['entity_id']}:triples",
                    "title": f"Entity {detail['name']} sample triples",
                    "content": " ; ".join(
                        f"{item['head_name']} -[{item['relation_name']}]-> {item['tail_name']}" for item in triple_items
                    ),
                    "source": "dbp15k",
                    "source_id": f"{detail['kg']}:{detail['entity_id']}",
                }
            )
        return chunks

    def _references_from_es_hits(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "source": "entity_linking_es",
                "rank": index,
                "title": hit.get("label") or "",
                "url": hit.get("link") or "",
                "score": round(_safe_float(hit.get("score")), 4),
            }
            for index, hit in enumerate(hits, start=1)
        ]

    def _chunks_from_es_hits(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        for index, hit in enumerate(hits, start=1):
            title = _normalize_text(hit.get("label"))
            link = _normalize_text(hit.get("link"))
            chunks.append(
                {
                    "chunk_id": f"es:{index}",
                    "title": title or f"Entity linking hit {index}",
                    "content": f"{title} | {link}",
                    "source": "entity_linking_es",
                    "source_id": link or f"es:{index}",
                    "score": round(_safe_float(hit.get("score")), 4),
                }
            )
        return chunks

    def _expand_alignment(self, entity: EntityRecord, top_k: int = 3) -> list[dict[str, Any]]:
        alignments: list[dict[str, Any]] = []
        matched = self.dataset.find_alignment(entity.kg, entity.entity_id, split="all")
        for split, aligned_entity in matched.items():
            alignments.append(
                self._format_alignment(
                    source=entity,
                    split=split,
                    target=aligned_entity,
                    score=1.0,
                    evidence="gold_alignment_split",
                )
            )

        if alignments or not self.config.enable_alignment_expansion:
            return alignments[:top_k]

        counterpart_kg: KgKey = "2" if entity.kg == "1" else "1"
        embedding_name = self.config.embedding_family
        try:
            source_ids, source_matrix = self.dataset.get_embedding_matrix(entity.kg, embedding_name=embedding_name)
            target_ids, target_matrix = self.dataset.get_embedding_matrix(counterpart_kg, embedding_name=embedding_name)
        except Exception:  # noqa: BLE001
            return []

        source_lookup = {int(entity_id): idx for idx, entity_id in enumerate(source_ids.tolist())}
        source_index = source_lookup.get(int(entity.entity_id))
        if source_index is None:
            return []

        vector = source_matrix[source_index]
        scores = target_matrix @ vector
        top_indices = np.argsort(scores)[::-1][:top_k]
        for rank_index in top_indices:
            target_id = int(target_ids[int(rank_index)])
            target_entity = self.dataset.get_entity(counterpart_kg, target_id)
            if target_entity is None:
                continue
            alignments.append(
                self._format_alignment(
                    source=entity,
                    split="predicted",
                    target=target_entity,
                    score=float(scores[int(rank_index)]),
                    evidence=f"raw_embedding:{embedding_name}",
                )
            )
        return alignments[:top_k]

    def get_entity_detail(
        self,
        entity_key: str,
        relation_limit: int | None = None,
        triple_limit: int | None = None,
        enable_alignment_expansion: bool | None = None,
    ) -> dict[str, Any]:
        parsed = self._parse_entity_key(entity_key)
        if parsed is None:
            raise KeyError(f"Invalid entity key: {entity_key}")
        kg, entity_id = parsed
        entity = self.dataset.get_entity(kg, entity_id)
        if entity is None:
            raise KeyError(f"Entity not found: {entity_key}")

        resolved_relation_limit = relation_limit or self.config.default_relation_limit
        resolved_triple_limit = triple_limit or self.config.default_triple_limit
        detail = self.dataset.describe_entity(
            kg=kg,
            entity_id=entity_id,
            relation_limit=resolved_relation_limit,
            triple_limit=resolved_triple_limit,
        )
        alignment_enabled = (
            self.config.enable_alignment_expansion
            if enable_alignment_expansion is None
            else bool(enable_alignment_expansion)
        )
        alignments = self._expand_alignment(entity, top_k=3) if alignment_enabled else []
        relationships = [
            {
                "relation_id": item.get("relation_id"),
                "relation_name": item.get("relation_name"),
                "match_score": 1.0,
                "sample_triple_count": item.get("count"),
                "description": f"Entity-local relation frequency for {entity.name}",
                "kg": kg,
                "kg_side": _kg_to_side(self.dataset_name, kg),
                "source_id": f"rel:{kg}:{item.get('relation_id')}",
            }
            for item in detail.get("relation_summary", [])
        ]
        return {
            "entity": self._format_entity(entity, match_source="entity_lookup"),
            "relationships": relationships,
            "triples": detail.get("triples", []),
            "alignments": alignments,
            "chunks": self._entity_summary_chunks(detail, resolved_relation_limit, resolved_triple_limit),
            "metadata": {
                "query_mode": "kg-rag",
                "entity_key": entity_key,
                "dataset": self.dataset_name,
                "kg_side": _kg_to_side(self.dataset_name, kg),
                "source": "kg-rag",
            },
        }

    def get_relation_detail(
        self,
        kg: str,
        relation_id: int,
        triple_limit: int | None = None,
    ) -> dict[str, Any]:
        relation = self.dataset.get_relation(kg, relation_id)
        if relation is None:
            raise KeyError(f"Relation not found: {kg}:{relation_id}")
        limit = triple_limit or self.config.default_triple_limit
        triples = self.dataset.search_triples_by_relation(kg=kg, relation_id=relation_id, limit=limit)
        rendered = [self.dataset.render_triple(triple) for triple in triples]
        return {
            "relation": self._format_relation(relation, sample_triple_count=len(rendered)),
            "triples": rendered,
            "chunks": (
                [
                    {
                        "chunk_id": f"relation:{kg}:{relation_id}",
                        "title": f"Relation {relation.name}",
                        "content": " ; ".join(
                            f"{item['head_name']} -[{item['relation_name']}]-> {item['tail_name']}"
                            for item in rendered
                        ),
                        "source": "dbp15k",
                        "source_id": f"rel:{kg}:{relation_id}",
                    }
                ]
                if rendered
                else []
            ),
            "metadata": {
                "query_mode": "kg-rag",
                "dataset": self.dataset_name,
                "kg_side": _kg_to_side(self.dataset_name, kg),
                "source": "kg-rag",
            },
        }

    def _detect_intent(self, query: str, requested: str = "auto") -> str:
        if requested and requested != "auto":
            return requested
        normalized = _normalize_text(query)
        lowered = normalized.casefold()
        if self._parse_relation_key(normalized):
            return "relation_lookup"
        if self._parse_entity_key(normalized):
            return "entity_lookup"
        if "三元组" in normalized or "triple" in lowered:
            return "triple_lookup"
        if "关系检索" in normalized or "relation search" in lowered:
            return "relation_search"
        if "关系" in normalized or "relation" in lowered:
            return "relation_search"
        return "hybrid"

    def _build_raw_result(
        self,
        answer: str,
        data: dict[str, Any],
        resolved_intent: str,
        processing_info: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "answer": answer,
            "data": json.loads(json.dumps(data, ensure_ascii=False)),
            "metadata": {
                "query_mode": "kg-rag",
                "detected_intent": resolved_intent,
                "processing_info": json.loads(json.dumps(processing_info, ensure_ascii=False)),
                "source": "kg-rag",
            },
        }

    def query(
        self,
        query: str,
        query_intent: str | None = None,
        retrieval_mode: str | None = None,
        top_k: int | None = None,
        relation_limit: int | None = None,
        triple_limit: int | None = None,
        enable_alignment_expansion: bool | None = None,
        kg_side: str | None = None,
        include_raw_result: bool = False,
    ) -> dict[str, Any]:
        resolved_intent = self._detect_intent(query, query_intent or self.config.default_query_intent)
        resolved_top_k = max(1, int(top_k or self.config.default_top_k))
        resolved_relation_limit = max(1, int(relation_limit or self.config.default_relation_limit))
        resolved_triple_limit = max(1, int(triple_limit or self.config.default_triple_limit))
        resolved_kg_side = kg_side or self.config.default_kg_side
        alignment_enabled = (
            self.config.enable_alignment_expansion
            if enable_alignment_expansion is None
            else bool(enable_alignment_expansion)
        )
        resolved_retrieval_mode = retrieval_mode or self.config.default_retrieval_mode

        data = {
            "entities": [],
            "relationships": [],
            "chunks": [],
            "references": [],
            "triples": [],
            "alignments": [],
        }
        processing_info: dict[str, Any] = {
            "dataset": self.dataset_name,
            "retrieval_mode": resolved_retrieval_mode,
            "fallbacks": [],
        }

        entity_link_hits, es_info = self._search_entity_linking(query, limit=resolved_top_k)
        processing_info.update(es_info)
        data["references"].extend(self._references_from_es_hits(entity_link_hits))
        data["chunks"].extend(self._chunks_from_es_hits(entity_link_hits[:3]))

        answer = "未命中结果。"
        if resolved_intent == "entity_lookup":
            parsed = self._parse_entity_key(query)
            if parsed is None:
                raise KeyError(f"Invalid entity key: {query}")
            detail = self.get_entity_detail(
                entity_key=_entity_key(parsed[0], parsed[1]),
                relation_limit=resolved_relation_limit,
                triple_limit=resolved_triple_limit,
                enable_alignment_expansion=alignment_enabled,
            )
            data["entities"] = [detail["entity"]]
            data["relationships"] = detail["relationships"]
            data["triples"] = detail["triples"]
            data["alignments"] = detail["alignments"]
            data["chunks"] = detail["chunks"] + data["chunks"]
            answer = f"已定位实体 {detail['entity']['entity_name']}，返回其关系、三元组和对齐扩展。"
        elif resolved_intent == "relation_lookup":
            parsed = self._parse_relation_key(query)
            if parsed is None:
                raise KeyError(f"Invalid relation key: {query}")
            detail = self.get_relation_detail(parsed[0], parsed[1], triple_limit=resolved_triple_limit)
            data["relationships"] = [detail["relation"]]
            data["triples"] = detail["triples"]
            data["chunks"] = detail["chunks"] + data["chunks"]
            answer = f"已定位关系 {detail['relation']['relation_name']}，返回样例三元组。"
        else:
            entity_matches = self._search_entities_local(query, kg_side=resolved_kg_side, limit=resolved_top_k)
            relation_matches = self._search_relations_local(
                query,
                kg_side=resolved_kg_side,
                limit=resolved_relation_limit,
            )

            prioritize_relations = resolved_retrieval_mode == "relation_first" or resolved_intent in {
                "relation_search",
                "triple_lookup",
            }

            if entity_matches and not prioritize_relations:
                top_entity, _ = entity_matches[0]
                detail = self.get_entity_detail(
                    entity_key=_entity_key(top_entity.kg, top_entity.entity_id),
                    relation_limit=resolved_relation_limit,
                    triple_limit=resolved_triple_limit,
                    enable_alignment_expansion=alignment_enabled,
                )
                data["entities"] = [self._format_entity(entity, score) for entity, score in entity_matches]
                data["relationships"] = detail["relationships"]
                data["triples"] = detail["triples"]
                data["alignments"] = detail["alignments"]
                data["chunks"] = detail["chunks"] + data["chunks"]
                answer = f"已按实体优先命中 {detail['entity']['entity_name']}，并补充关系、三元组与对齐证据。"
            elif entity_matches:
                data["entities"] = [self._format_entity(entity, score) for entity, score in entity_matches]

            if relation_matches and (prioritize_relations or not data["relationships"]):
                rendered_relations: list[dict[str, Any]] = []
                triple_payloads: list[dict[str, Any]] = list(data["triples"])
                triple_keys = {_triple_key(item) for item in triple_payloads}
                for relation, score in relation_matches[:resolved_relation_limit]:
                    triples = self.dataset.search_triples_by_relation(
                        kg=relation.kg,
                        relation_id=relation.relation_id,
                        limit=max(3, min(resolved_triple_limit, 10)),
                    )
                    rendered_relations.append(
                        self._format_relation(relation, match_score=score, sample_triple_count=len(triples))
                    )
                    for triple in triples:
                        rendered = self.dataset.render_triple(triple)
                        if _triple_key(rendered) not in triple_keys:
                            triple_keys.add(_triple_key(rendered))
                            triple_payloads.append(rendered)
                if rendered_relations:
                    data["relationships"] = rendered_relations
                data["triples"] = triple_payloads[:resolved_triple_limit]
                if prioritize_relations:
                    answer = "已按关系优先返回关系匹配与样例三元组。"
                elif not data["entities"]:
                    answer = "未命中明确实体，返回关系检索结果与样例三元组。"

            if entity_matches and prioritize_relations and not data["alignments"]:
                top_entity, _ = entity_matches[0]
                detail = self.get_entity_detail(
                    entity_key=_entity_key(top_entity.kg, top_entity.entity_id),
                    relation_limit=resolved_relation_limit,
                    triple_limit=resolved_triple_limit,
                    enable_alignment_expansion=alignment_enabled,
                )
                data["alignments"] = detail["alignments"]
                if not data["chunks"]:
                    data["chunks"] = detail["chunks"]

            if not data["entities"] and not data["relationships"]:
                processing_info["fallbacks"].append("dbp15k_entity_name_search_empty")
                answer = "未命中明确实体或关系，返回实体链接检索证据。"

        result = {
            "answer": answer,
            "data": data,
            "metadata": {
                "query_mode": "kg-rag",
                "detected_intent": resolved_intent,
                "processing_info": processing_info,
                "source": "kg-rag",
            },
        }
        if include_raw_result:
            result["metadata"]["raw_result"] = self._build_raw_result(answer, data, resolved_intent, processing_info)
        return result

    def list_relations(self, kg_side: str | None = None) -> list[dict[str, Any]]:
        """列出所有关系类型及其三元组数量。"""
        resolved_kg_side = kg_side or self.config.default_kg_side
        result: list[dict[str, Any]] = []
        for kg in _side_to_kgs(self.dataset_name, resolved_kg_side):
            for relation_id, relation in self.dataset.relations[kg].items():
                triples = self.dataset._relation_triples.get(kg, {}).get(relation_id, [])
                result.append({
                    "relation_id": relation_id,
                    "name": relation.name,
                    "kg": kg,
                    "kg_side": _kg_to_side(self.dataset_name, kg),
                    "triple_count": len(triples),
                })
        result.sort(key=lambda r: -r["triple_count"])
        return result

    def build_subgraph(
        self,
        node_label: str = "*",
        max_nodes: int = 50,
        max_depth: int = 1,
        enable_alignment_expansion: bool | None = None,
        kg_side: str | None = None,
        relation_id: int | None = None,
        relation_kg: str | None = None,
    ) -> dict[str, Any]:
        alignment_enabled = (
            self.config.enable_alignment_expansion
            if enable_alignment_expansion is None
            else bool(enable_alignment_expansion)
        )
        resolved_kg_side = kg_side or self.config.default_kg_side
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[str, dict[str, Any]] = {}

        import re as _re
        from urllib.parse import unquote

        _BAD_NAME_RE = _re.compile(r"^\[?\d+\]?$")

        def _is_bad_name(name: str) -> bool:
            return not name or _BAD_NAME_RE.match(name.strip()) is not None

        def _clean_name(raw: str) -> str:
            """将 URL 或坏名称转为可读实体名。"""
            if not raw:
                return ""
            stripped = raw.strip()
            # URL → 取最后一段路径并解码
            if stripped.startswith("http://") or stripped.startswith("https://"):
                path = stripped.rstrip("/").rsplit("/", 1)[-1]
                # 去掉 ?title= 参数格式
                if "?title=" in path:
                    path = path.split("?title=", 1)[-1]
                decoded = unquote(path).replace("_", " ")
                return decoded or stripped
            return stripped

        def _display_name(entity: EntityRecord) -> str:
            cleaned = _clean_name(entity.name)
            if _is_bad_name(cleaned):
                return f"entity_{entity.kg}:{entity.entity_id}"
            return cleaned

        def add_entity_node(entity: EntityRecord) -> None:
            node_id = f"{entity.kg}:{entity.entity_id}"
            display = _display_name(entity)
            link = entity.name if entity.name.startswith("http") else ""
            nodes[node_id] = {
                "id": node_id,
                "name": display,
                "original_id": node_id,
                "type": "entity",
                "labels": ["entity", _kg_to_side(self.dataset_name, entity.kg)],
                "properties": {
                    "entity_id": entity.entity_id,
                    "kg": entity.kg,
                    "kg_side": _kg_to_side(self.dataset_name, entity.kg),
                    **({"link": link} if link else {}),
                },
                "normalized": {"name": display, "type": "entity", "source": "dbp15k"},
                "graph_type": "kg-rag",
            }

        # --- 按关系类型过滤模式 ---
        if relation_id is not None:
            r_kg = relation_kg or ("1" if resolved_kg_side in ("auto", "zh", "1") else "2")
            triples = self.dataset._relation_triples.get(r_kg, {}).get(relation_id, [])
            relation = self.dataset.get_relation(r_kg, relation_id)
            rel_name = relation.name if relation else f"relation_{relation_id}"
            for triple in triples[:max_nodes]:
                head = self.dataset.get_entity(r_kg, triple.head_id)
                tail = self.dataset.get_entity(r_kg, triple.tail_id)
                if head is None or tail is None:
                    continue
                add_entity_node(head)
                add_entity_node(tail)
                edge_id = f"rel:{r_kg}:{triple.head_id}:{triple.relation_id}:{triple.tail_id}"
                edges[edge_id] = {
                    "id": edge_id,
                    "source_id": f"{r_kg}:{triple.head_id}",
                    "target_id": f"{r_kg}:{triple.tail_id}",
                    "type": rel_name,
                    "properties": {"relation_id": triple.relation_id, "relation_name": rel_name, "kg": r_kg},
                    "normalized": {"type": rel_name, "direction": "directed"},
                }
            trimmed_nodes = list(nodes.values())[:max_nodes]
            valid_ids = {n["id"] for n in trimmed_nodes}
            trimmed_edges = [e for e in edges.values() if e["source_id"] in valid_ids and e["target_id"] in valid_ids]
            return {"nodes": trimmed_nodes, "edges": trimmed_edges}

        # --- 正常种子模式 ---
        seed_entities: list[EntityRecord] = []
        parsed = self._parse_entity_key(node_label)
        if node_label == "*":
            counter: list[tuple[int, EntityRecord]] = []
            for kg in _side_to_kgs(self.dataset_name, resolved_kg_side):
                for entity_id, relation_counts in self.dataset._entity_relation_counts[kg].items():
                    entity = self.dataset.get_entity(kg, entity_id)
                    if entity is not None and not _is_bad_name(_clean_name(entity.name)):
                        counter.append((sum(relation_counts.values()), entity))
            counter.sort(key=lambda item: (-item[0], item[1].kg, item[1].entity_id))
            seed_entities = [entity for _, entity in counter[: max(1, min(max_nodes, 10))]]
        elif parsed is not None:
            entity = self.dataset.get_entity(parsed[0], parsed[1])
            if entity is not None:
                seed_entities = [entity]
        else:
            seed_entities = [
                entity
                for entity, _ in self._search_entities_local(node_label, resolved_kg_side, limit=min(max_nodes, 5))
            ]

        for entity in seed_entities:
            add_entity_node(entity)
            triples = self.dataset.get_triples_for_entity(entity.kg, entity.entity_id)[:max_nodes]
            for triple in triples:
                head = self.dataset.get_entity(entity.kg, triple.head_id)
                tail = self.dataset.get_entity(entity.kg, triple.tail_id)
                relation = self.dataset.get_relation(entity.kg, triple.relation_id)
                if head is None or tail is None or relation is None:
                    continue
                add_entity_node(head)
                add_entity_node(tail)
                edge_id = f"rel:{entity.kg}:{triple.head_id}:{triple.relation_id}:{triple.tail_id}"
                edges[edge_id] = {
                    "id": edge_id,
                    "source_id": f"{entity.kg}:{triple.head_id}",
                    "target_id": f"{entity.kg}:{triple.tail_id}",
                    "type": relation.name,
                    "properties": {
                        "relation_id": triple.relation_id,
                        "relation_name": relation.name,
                        "kg": entity.kg,
                    },
                    "normalized": {"type": relation.name, "direction": "directed"},
                }
            if alignment_enabled and max_depth > 0:
                for alignment in self._expand_alignment(entity, top_k=1):
                    target = alignment["target_entity"]
                    target_record = self.dataset.get_entity(target["kg"], target["entity_id"])
                    if target_record is None:
                        continue
                    add_entity_node(target_record)
                    edge_id = f"align:{entity.kg}:{entity.entity_id}:{target_record.kg}:{target_record.entity_id}"
                    edges[edge_id] = {
                        "id": edge_id,
                        "source_id": f"{entity.kg}:{entity.entity_id}",
                        "target_id": f"{target_record.kg}:{target_record.entity_id}",
                        "type": "aligned_with",
                        "properties": {
                            "score": alignment["score"],
                            "split": alignment["split"],
                            "evidence": alignment["evidence"],
                        },
                        "normalized": {"type": "aligned_with", "direction": "undirected"},
                    }

        trimmed_nodes = list(nodes.values())[:max_nodes]
        valid_ids = {node["id"] for node in trimmed_nodes}
        trimmed_edges = [
            edge for edge in edges.values() if edge["source_id"] in valid_ids and edge["target_id"] in valid_ids
        ]
        return {"nodes": trimmed_nodes, "edges": trimmed_edges}

    def generate_benchmark(
        self,
        output_path: str | Path,
        count: int = 40,
        seed: int = 42,
    ) -> dict[str, Any]:
        rng = random.Random(seed)
        pairs = self.dataset.get_alignment_pairs("test")
        if not pairs:
            raise RuntimeError("No alignment pairs available for benchmark generation")

        sampled_pairs = list(pairs)
        rng.shuffle(sampled_pairs)
        records: list[dict[str, Any]] = []
        query_types = ["entity_lookup", "hybrid", "relation_search", "triple_lookup"]

        for index in range(count):
            pair = sampled_pairs[index % len(sampled_pairs)]
            left = self.dataset.get_entity("1", pair.left_id)
            right = self.dataset.get_entity("2", pair.right_id)
            if left is None or right is None:
                continue

            relation_counts = self.dataset.get_relation_counts_for_entity(left.kg, left.entity_id)
            top_relation = relation_counts[0][0] if relation_counts else None
            sample_triples = [
                self.dataset.render_triple(triple)
                for triple in self.dataset.get_triples_for_entity(left.kg, left.entity_id)[:3]
            ]
            query_type = query_types[index % len(query_types)]
            if query_type == "entity_lookup":
                query_text = _entity_key(left.kg, left.entity_id)
            elif query_type == "hybrid":
                query_text = left.name
            elif query_type == "relation_search" and top_relation is not None:
                query_text = top_relation.name
            else:
                relation_part = top_relation.name if top_relation is not None else ""
                query_text = f"{left.name} {relation_part} 三元组".strip()

            records.append(
                {
                    "query": query_text,
                    "query_type": query_type,
                    "query_side": _kg_to_side(self.dataset_name, left.kg),
                    "gold_entities": [self._format_entity(left)],
                    "gold_relations": (
                        [self._format_relation(top_relation, sample_triple_count=len(sample_triples))]
                        if top_relation is not None
                        else []
                    ),
                    "gold_triples": sample_triples,
                    "gold_alignments": [self._format_alignment(left, "test", right, 1.0, "gold_alignment_split")],
                    "gold_answer": f"{left.name} aligned with {right.name}",
                    "metadata": {"seed": seed, "benchmark_index": index},
                }
            )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {
            "benchmark_file": str(output),
            "question_count": len(records),
            "has_gold_chunks": False,
            "has_gold_answers": True,
            "created_at": _utc_now(),
        }

    def run_benchmark_eval(
        self,
        benchmark_path: str | Path,
        output_dir: str | Path,
        top_k: int | None = None,
        relation_limit: int | None = None,
        triple_limit: int | None = None,
        enable_alignment_expansion: bool | None = None,
    ) -> dict[str, Any]:
        benchmark_file = Path(benchmark_path)
        if not benchmark_file.exists():
            raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")

        rows = [json.loads(line) for line in benchmark_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise RuntimeError("Benchmark file is empty")

        entity_ranks: list[int | None] = []
        relation_ranks: list[int | None] = []
        alignment_ranks: list[int | None] = []
        triple_scores: list[tuple[float, float, float]] = []
        details: list[dict[str, Any]] = []

        for row in rows:
            result = self.query(
                row["query"],
                query_intent=row.get("query_type") if row.get("query_type") != "hybrid" else "auto",
                top_k=top_k,
                relation_limit=relation_limit,
                triple_limit=triple_limit,
                enable_alignment_expansion=enable_alignment_expansion,
                kg_side=row.get("query_side", "auto"),
            )
            entity_ranks.append(self._rank_structured(result["data"]["entities"], row.get("gold_entities", []), "source_id"))
            relation_ranks.append(
                self._rank_structured(result["data"]["relationships"], row.get("gold_relations", []), "source_id")
            )
            alignment_ranks.append(
                self._rank_structured(result["data"]["alignments"], row.get("gold_alignments", []), "target_entity")
            )
            precision, recall, f1 = self._triple_metrics(result["data"]["triples"], row.get("gold_triples", []))
            triple_scores.append((precision, recall, f1))
            details.append(
                {
                    "query": row["query"],
                    "gold": row,
                    "result": result,
                    "metrics": {
                        "entity_rank": entity_ranks[-1],
                        "relation_rank": relation_ranks[-1],
                        "alignment_rank": alignment_ranks[-1],
                        "triple_precision": precision,
                        "triple_recall": recall,
                        "triple_f1": f1,
                    },
                }
            )

        summary = self._benchmark_summary(entity_ranks, relation_ranks, alignment_ranks, triple_scores)
        artifacts = self._write_eval_artifacts(
            output_dir=output_dir,
            summary=summary,
            details=details,
            title="kg-rag benchmark evaluation",
        )
        summary["artifacts"] = artifacts
        return summary

    def run_official_comparison(self, output_dir: str | Path, split: str = "test", device: str = "cpu") -> dict[str, Any]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        summary: dict[str, Any] = {"dataset": self.dataset_name, "split": split, "generated_at": _utc_now()}
        summary["raw_labse"] = evaluate_raw_alignment(self.dataset, split=split, embedding_name="labse").to_dict()

        if self.config.alignment_model_path and Path(self.config.alignment_model_path).exists():
            summary["labse_neighbor"] = evaluate_final_model_alignment(
                self.dataset,
                model_path=self.config.alignment_model_path,
                split=split,
                device=device,
                embedding_name="labse",
            ).to_dict()
        else:
            summary["labse_neighbor"] = {"missing": str(self.config.alignment_model_path or "")}

        try:
            summary["raw_bge_m3"] = evaluate_raw_alignment(
                self.dataset,
                split=split,
                embedding_name="bge_m3",
            ).to_dict()
        except Exception as exc:  # noqa: BLE001
            summary["raw_bge_m3"] = {"missing": str(exc)}

        if self.config.bge_alignment_model_path and Path(self.config.bge_alignment_model_path).exists():
            try:
                summary["bge_m3_neighbor"] = evaluate_final_model_alignment(
                    self.dataset,
                    model_path=self.config.bge_alignment_model_path,
                    split=split,
                    device=device,
                    embedding_name="bge_m3",
                ).to_dict()
            except Exception as exc:  # noqa: BLE001
                summary["bge_m3_neighbor"] = {"missing": str(exc)}
        else:
            summary["bge_m3_neighbor"] = {"missing": str(self.config.bge_alignment_model_path or "")}

        summary["delta_labse"] = self._delta_metrics(summary["raw_labse"], summary["labse_neighbor"])
        summary["delta_bge_m3"] = self._delta_metrics(summary["raw_bge_m3"], summary["bge_m3_neighbor"])
        artifacts = self._write_eval_artifacts(
            output_dir=output,
            summary=summary,
            details=[],
            title="kg-rag official comparison",
        )
        summary["artifacts"] = artifacts
        return summary

    def _rank_structured(
        self,
        predicted: list[dict[str, Any]],
        gold: list[dict[str, Any]],
        key: str,
    ) -> int | None:
        if not predicted or not gold:
            return None
        gold_values: set[str] = set()
        for item in gold:
            if key == "target_entity":
                target = item.get("target_entity") or {}
                gold_values.add(str(target.get("source_id") or target.get("detail_route") or ""))
            else:
                gold_values.add(str(item.get(key) or ""))
        gold_values.discard("")
        if not gold_values:
            return None

        for index, item in enumerate(predicted, start=1):
            if key == "target_entity":
                target = item.get("target_entity") or {}
                value = str(target.get("source_id") or target.get("detail_route") or "")
            else:
                value = str(item.get(key) or "")
            if value in gold_values:
                return index
        return None

    def _triple_metrics(self, predicted: list[dict[str, Any]], gold: list[dict[str, Any]]) -> tuple[float, float, float]:
        predicted_set = {_triple_key(item) for item in predicted}
        gold_set = {_triple_key(item) for item in gold}
        if not predicted_set and not gold_set:
            return 1.0, 1.0, 1.0
        if not predicted_set:
            return 0.0, 0.0, 0.0
        intersection = predicted_set & gold_set
        precision = len(intersection) / len(predicted_set) if predicted_set else 0.0
        recall = len(intersection) / len(gold_set) if gold_set else 0.0
        if precision + recall == 0:
            return 0.0, 0.0, 0.0
        f1 = 2 * precision * recall / (precision + recall)
        return round(precision, 4), round(recall, 4), round(f1, 4)

    def _hits_and_mrr(self, ranks: list[int | None]) -> dict[str, float]:
        valid = len(ranks)
        if valid == 0:
            return {"mrr": 0.0, "hits@1": 0.0, "hits@5": 0.0, "hits@10": 0.0}
        reciprocal = 0.0
        hits = Counter()
        for rank in ranks:
            if rank is None:
                continue
            reciprocal += 1.0 / rank
            for threshold in (1, 5, 10):
                if rank <= threshold:
                    hits[threshold] += 1
        return {
            "mrr": round(reciprocal / valid, 4),
            "hits@1": round(hits[1] / valid, 4),
            "hits@5": round(hits[5] / valid, 4),
            "hits@10": round(hits[10] / valid, 4),
        }

    def _benchmark_summary(
        self,
        entity_ranks: list[int | None],
        relation_ranks: list[int | None],
        alignment_ranks: list[int | None],
        triple_scores: list[tuple[float, float, float]],
    ) -> dict[str, Any]:
        if triple_scores:
            triple_precision = round(sum(item[0] for item in triple_scores) / len(triple_scores), 4)
            triple_recall = round(sum(item[1] for item in triple_scores) / len(triple_scores), 4)
            triple_f1 = round(sum(item[2] for item in triple_scores) / len(triple_scores), 4)
        else:
            triple_precision = 0.0
            triple_recall = 0.0
            triple_f1 = 0.0

        summary = {
            "dataset": self.dataset_name,
            "generated_at": _utc_now(),
            "question_count": len(entity_ranks),
            "entity": self._hits_and_mrr(entity_ranks),
            "relation": self._hits_and_mrr(relation_ranks),
            "alignment": self._hits_and_mrr(alignment_ranks),
            "triple_precision": triple_precision,
            "triple_recall": triple_recall,
            "triple_f1": triple_f1,
        }
        available = [
            summary["entity"]["mrr"],
            summary["relation"]["mrr"],
            summary["alignment"]["mrr"],
            triple_f1,
        ]
        summary["overall_score"] = round(sum(available) / len(available), 4)
        return summary

    def _delta_metrics(self, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        if "mrr" not in before or "mrr" not in after:
            return {}
        return {
            "mrr": round(_safe_float(after.get("mrr")) - _safe_float(before.get("mrr")), 4),
            "hits@1": round(_safe_float(after.get("hits@1")) - _safe_float(before.get("hits@1")), 4),
            "hits@5": round(_safe_float(after.get("hits@5")) - _safe_float(before.get("hits@5")), 4),
            "hits@10": round(_safe_float(after.get("hits@10")) - _safe_float(before.get("hits@10")), 4),
        }

    def _write_eval_artifacts(
        self,
        output_dir: str | Path,
        summary: dict[str, Any],
        details: list[dict[str, Any]],
        title: str,
    ) -> dict[str, str]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        summary_path = output / "summary.json"
        details_path = output / "details.jsonl"
        report_path = output / "report.md"

        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        with open(details_path, "w", encoding="utf-8") as handle:
            for item in details:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(self._render_summary_markdown(summary, title))

        return {
            "summary_json": str(summary_path),
            "details_jsonl": str(details_path),
            "report_md": str(report_path),
        }

    def _render_summary_markdown(self, summary: dict[str, Any], title: str) -> str:
        lines = [
            f"# {title}",
            "",
            f"- dataset: `{summary.get('dataset', self.dataset_name)}`",
            f"- generated_at: `{summary.get('generated_at', _utc_now())}`",
            "",
        ]
        if "entity" in summary:
            lines.extend(
                [
                    "## benchmark metrics",
                    "",
                    f"- entity_mrr: `{summary['entity']['mrr']}`",
                    f"- entity_hits@1: `{summary['entity']['hits@1']}`",
                    f"- relation_mrr: `{summary['relation']['mrr']}`",
                    f"- alignment_mrr: `{summary['alignment']['mrr']}`",
                    f"- triple_f1: `{summary['triple_f1']}`",
                    f"- overall_score: `{summary['overall_score']}`",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "## official comparison",
                    "",
                    f"- raw_labse_mrr: `{summary.get('raw_labse', {}).get('mrr', 'n/a')}`",
                    f"- labse_neighbor_mrr: `{summary.get('labse_neighbor', {}).get('mrr', 'n/a')}`",
                    f"- raw_bge_m3_mrr: `{summary.get('raw_bge_m3', {}).get('mrr', 'n/a')}`",
                    f"- bge_m3_neighbor_mrr: `{summary.get('bge_m3_neighbor', {}).get('mrr', 'n/a')}`",
                    "",
                ]
            )
        return "\n".join(lines) + "\n"
