from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np


KgKey = Literal["1", "2"]
SplitKey = Literal["test", "valid", "ref_ent_ids"]
EmbeddingKey = Literal["labse", "bge_m3"]

EMBEDDING_FILE_PREFIX = {
    "labse": "raw_LaBSE_emb",
    "bge_m3": "raw_BGE_M3_emb",
}


@dataclass(frozen=True)
class EntityRecord:
    kg: KgKey
    entity_id: int
    name: str


@dataclass(frozen=True)
class RelationRecord:
    kg: KgKey
    relation_id: int
    name: str


@dataclass(frozen=True)
class TripleRecord:
    kg: KgKey
    head_id: int
    relation_id: int
    tail_id: int


@dataclass(frozen=True)
class AlignmentPair:
    left_id: int
    right_id: int


def _split_id_and_text(line: str) -> tuple[int, str]:
    stripped = line.lstrip("\ufeff").rstrip("\n")
    if not stripped:
        raise ValueError("empty line")
    parts = stripped.split(maxsplit=1)
    idx = int(parts[0])
    text = parts[1].strip() if len(parts) > 1 else ""
    return idx, text


def _load_id_mapping(path: Path, kg: KgKey, kind: str) -> dict[int, EntityRecord | RelationRecord]:
    records: dict[int, EntityRecord | RelationRecord] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            idx, text = _split_id_and_text(raw_line)
            if kind == "entity":
                records[idx] = EntityRecord(kg=kg, entity_id=idx, name=text)
            else:
                records[idx] = RelationRecord(kg=kg, relation_id=idx, name=text)
    return records


def _load_triples(path: Path, kg: KgKey) -> list[TripleRecord]:
    triples: list[TripleRecord] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            values = raw_line.lstrip("\ufeff").split()
            head_id, relation_id, tail_id = (int(value) for value in values)
            triples.append(TripleRecord(kg=kg, head_id=head_id, relation_id=relation_id, tail_id=tail_id))
    return triples


def _load_alignment_pairs(path: Path) -> list[AlignmentPair]:
    pairs: list[AlignmentPair] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            values = raw_line.lstrip("\ufeff").split()
            left_id, right_id = (int(value) for value in values)
            pairs.append(AlignmentPair(left_id=left_id, right_id=right_id))
    return pairs


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


class DBP15KDataset:
    def __init__(self, dataset_dir: Path, dataset_name: str | None = None) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.dataset_name = dataset_name or self.dataset_dir.name

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"DBP15K dataset directory not found: {self.dataset_dir}")

        self.entities: dict[KgKey, dict[int, EntityRecord]] = {
            "1": _load_id_mapping(self.dataset_dir / "cleaned_ent_ids_1", "1", "entity"),
            "2": _load_id_mapping(self.dataset_dir / "cleaned_ent_ids_2", "2", "entity"),
        }
        self.relations: dict[KgKey, dict[int, RelationRecord]] = {
            "1": _load_id_mapping(self.dataset_dir / "cleaned_rel_ids_1", "1", "relation"),
            "2": _load_id_mapping(self.dataset_dir / "cleaned_rel_ids_2", "2", "relation"),
        }
        self.triples: dict[KgKey, list[TripleRecord]] = {
            "1": _load_triples(self.dataset_dir / "triples_1", "1"),
            "2": _load_triples(self.dataset_dir / "triples_2", "2"),
        }
        self.alignments: dict[SplitKey, list[AlignmentPair]] = {
            "test": _load_alignment_pairs(self.dataset_dir / "test"),
            "valid": _load_alignment_pairs(self.dataset_dir / "valid"),
            "ref_ent_ids": _load_alignment_pairs(self.dataset_dir / "ref_ent_ids"),
        }

        self._entity_triples: dict[KgKey, dict[str, dict[int, list[TripleRecord]]]] = {
            "1": self._build_entity_triple_index("1"),
            "2": self._build_entity_triple_index("2"),
        }
        self._relation_triples: dict[KgKey, dict[int, list[TripleRecord]]] = {
            "1": self._build_relation_index("1"),
            "2": self._build_relation_index("2"),
        }
        self._entity_relation_counts: dict[KgKey, dict[int, Counter[int]]] = {
            "1": self._build_entity_relation_count("1"),
            "2": self._build_entity_relation_count("2"),
        }
        self._alignment_lookup: dict[SplitKey, dict[KgKey, dict[int, int]]] = self._build_alignment_lookup()
        self._embedding_cache: dict[tuple[KgKey, str], dict[int, np.ndarray]] = {}
        self._embedding_matrix_cache: dict[tuple[KgKey, str], tuple[np.ndarray, np.ndarray]] = {}
        self._neighbor_id_cache: dict[tuple[KgKey, int, str], tuple[list[int], dict[int, list[int]]]] = {}

    def _build_entity_triple_index(self, kg: KgKey) -> dict[str, dict[int, list[TripleRecord]]]:
        head_index: dict[int, list[TripleRecord]] = defaultdict(list)
        tail_index: dict[int, list[TripleRecord]] = defaultdict(list)
        both_index: dict[int, list[TripleRecord]] = defaultdict(list)
        for triple in self.triples[kg]:
            head_index[triple.head_id].append(triple)
            tail_index[triple.tail_id].append(triple)
            both_index[triple.head_id].append(triple)
            if triple.tail_id != triple.head_id:
                both_index[triple.tail_id].append(triple)
        return {"head": dict(head_index), "tail": dict(tail_index), "both": dict(both_index)}

    def _build_relation_index(self, kg: KgKey) -> dict[int, list[TripleRecord]]:
        relation_index: dict[int, list[TripleRecord]] = defaultdict(list)
        for triple in self.triples[kg]:
            relation_index[triple.relation_id].append(triple)
        return dict(relation_index)

    def _build_entity_relation_count(self, kg: KgKey) -> dict[int, Counter[int]]:
        relation_counter: dict[int, Counter[int]] = defaultdict(Counter)
        for triple in self.triples[kg]:
            relation_counter[triple.head_id][triple.relation_id] += 1
            relation_counter[triple.tail_id][triple.relation_id] += 1
        return dict(relation_counter)

    def _build_alignment_lookup(self) -> dict[SplitKey, dict[KgKey, dict[int, int]]]:
        lookup: dict[SplitKey, dict[KgKey, dict[int, int]]] = {}
        for split_name, pairs in self.alignments.items():
            left_lookup: dict[int, int] = {}
            right_lookup: dict[int, int] = {}
            for pair in pairs:
                left_lookup[pair.left_id] = pair.right_id
                right_lookup[pair.right_id] = pair.left_id
            lookup[split_name] = {"1": left_lookup, "2": right_lookup}
        return lookup

    def get_entity(self, kg: KgKey, entity_id: int) -> EntityRecord | None:
        return self.entities[kg].get(int(entity_id))

    def get_relation(self, kg: KgKey, relation_id: int) -> RelationRecord | None:
        return self.relations[kg].get(int(relation_id))

    def get_alignment_pairs(self, split: SplitKey) -> list[AlignmentPair]:
        return list(self.alignments[split])

    def find_alignment(self, kg: KgKey, entity_id: int, split: SplitKey | Literal["all"] = "all") -> dict[str, EntityRecord]:
        entity_id = int(entity_id)
        counterpart_kg: KgKey = "2" if kg == "1" else "1"
        splits: Iterable[SplitKey]
        if split == "all":
            splits = ("test", "valid", "ref_ent_ids")
        else:
            splits = (split,)

        results: dict[str, EntityRecord] = {}
        for split_name in splits:
            counterpart_id = self._alignment_lookup[split_name][kg].get(entity_id)
            if counterpart_id is None:
                continue
            counterpart = self.get_entity(counterpart_kg, counterpart_id)
            if counterpart is not None:
                results[split_name] = counterpart
        return results

    def get_triples_for_entity(
        self,
        kg: KgKey,
        entity_id: int,
        direction: Literal["head", "tail", "both"] = "both",
        relation_id: int | None = None,
    ) -> list[TripleRecord]:
        triples = list(self._entity_triples[kg][direction].get(int(entity_id), []))
        if relation_id is None:
            return triples
        return [triple for triple in triples if triple.relation_id == int(relation_id)]

    def get_relation_counts_for_entity(self, kg: KgKey, entity_id: int) -> list[tuple[RelationRecord | None, int]]:
        relation_counter = self._entity_relation_counts[kg].get(int(entity_id), Counter())
        sorted_pairs = sorted(relation_counter.items(), key=lambda item: (-item[1], item[0]))
        return [(self.get_relation(kg, relation_id), count) for relation_id, count in sorted_pairs]

    def search_relations(self, kg: KgKey, query: str, limit: int = 20) -> list[RelationRecord]:
        needle = query.casefold().strip()
        if not needle:
            return []

        if needle.isdigit():
            exact = self.get_relation(kg, int(needle))
            return [exact] if exact is not None else []

        scored: list[tuple[int, RelationRecord]] = []
        for relation in self.relations[kg].values():
            haystack = relation.name.casefold()
            if needle == haystack:
                scored.append((0, relation))
            elif haystack.startswith(needle):
                scored.append((1, relation))
            elif needle in haystack:
                scored.append((2, relation))
        scored.sort(key=lambda item: (item[0], item[1].relation_id))
        return [record for _, record in scored[:limit]]

    def search_triples_by_relation(
        self,
        kg: KgKey,
        relation_id: int | None = None,
        relation_query: str | None = None,
        entity_id: int | None = None,
        limit: int = 20,
    ) -> list[TripleRecord]:
        triples: list[TripleRecord] = []
        relation_ids: list[int] = []
        if relation_id is not None:
            relation_ids = [int(relation_id)]
        elif relation_query:
            relation_ids = [record.relation_id for record in self.search_relations(kg, relation_query, limit=limit)]
        else:
            raise ValueError("Either relation_id or relation_query must be provided")

        for current_relation_id in relation_ids:
            triples.extend(self._relation_triples[kg].get(current_relation_id, []))

        if entity_id is not None:
            entity_id = int(entity_id)
            triples = [
                triple
                for triple in triples
                if triple.head_id == entity_id or triple.tail_id == entity_id
            ]

        triples.sort(key=lambda triple: (triple.head_id, triple.tail_id, triple.relation_id))
        return triples[:limit]

    def render_triple(self, triple: TripleRecord) -> dict[str, object]:
        head = self.get_entity(triple.kg, triple.head_id)
        relation = self.get_relation(triple.kg, triple.relation_id)
        tail = self.get_entity(triple.kg, triple.tail_id)
        return {
            "kg": triple.kg,
            "head_id": triple.head_id,
            "head_name": head.name if head else "",
            "relation_id": triple.relation_id,
            "relation_name": relation.name if relation else "",
            "tail_id": triple.tail_id,
            "tail_name": tail.name if tail else "",
        }

    def describe_entity(
        self,
        kg: KgKey,
        entity_id: int,
        relation_limit: int = 10,
        triple_limit: int = 10,
    ) -> dict[str, object]:
        entity = self.get_entity(kg, entity_id)
        if entity is None:
            raise KeyError(f"Entity {entity_id} not found in KG{kg}")

        relation_summary = []
        for relation, count in self.get_relation_counts_for_entity(kg, entity_id)[:relation_limit]:
            relation_summary.append(
                {
                    "relation_id": relation.relation_id if relation else None,
                    "relation_name": relation.name if relation else "",
                    "count": count,
                }
            )

        triples = [self.render_triple(triple) for triple in self.get_triples_for_entity(kg, entity_id)[:triple_limit]]
        alignments = {
            split_name: {
                "entity_id": record.entity_id,
                "name": record.name,
                "kg": record.kg,
            }
            for split_name, record in self.find_alignment(kg, entity_id, split="all").items()
        }

        return {
            "dataset": self.dataset_name,
            "kg": kg,
            "entity_id": entity.entity_id,
            "name": entity.name,
            "alignments": alignments,
            "relation_summary": relation_summary,
            "triples": triples,
        }

    def _resolve_embedding_prefix(self, embedding_name: str) -> str:
        return EMBEDDING_FILE_PREFIX.get(embedding_name, embedding_name)

    def _resolve_embedding_path(self, kg: KgKey, embedding_name: str = "labse") -> Path:
        prefix = self._resolve_embedding_prefix(embedding_name)
        return self.dataset_dir / f"{prefix}_{kg}.pkl"

    def load_raw_embeddings(self, kg: KgKey, embedding_name: str = "labse") -> dict[int, np.ndarray]:
        cache_key = (kg, embedding_name)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        embedding_path = self._resolve_embedding_path(kg=kg, embedding_name=embedding_name)
        if not embedding_path.exists():
            raise FileNotFoundError(f"Raw embedding file not found: {embedding_path}")

        with open(embedding_path, "rb") as handle:
            payload = pickle.load(handle)

        embeddings: dict[int, np.ndarray] = {}
        for key, value in payload.items():
            vector = np.asarray(value, dtype=np.float32)
            if vector.ndim == 2 and vector.shape[0] == 1:
                vector = vector[0]
            elif vector.ndim != 1:
                vector = vector.reshape(-1)
            embeddings[int(key)] = vector

        self._embedding_cache[cache_key] = embeddings
        return embeddings

    def get_embedding_matrix(self, kg: KgKey, embedding_name: str = "labse") -> tuple[np.ndarray, np.ndarray]:
        cache_key = (kg, embedding_name)
        if cache_key in self._embedding_matrix_cache:
            return self._embedding_matrix_cache[cache_key]

        embeddings = self.load_raw_embeddings(kg, embedding_name=embedding_name)
        ordered_ids = np.asarray(sorted(embeddings.keys()), dtype=np.int64)
        matrix = np.stack([_normalize_vector(embeddings[entity_id]) for entity_id in ordered_ids]).astype(np.float32)
        self._embedding_matrix_cache[cache_key] = (ordered_ids, matrix)
        return ordered_ids, matrix

    def get_embedding_dim(self, kg: KgKey, embedding_name: str = "labse") -> int:
        embeddings = self.load_raw_embeddings(kg, embedding_name=embedding_name)
        if not embeddings:
            raise RuntimeError(f"No embeddings available for KG{kg}")
        first_vector = next(iter(embeddings.values()))
        return int(first_vector.shape[0])

    def build_neighbor_ids(
        self,
        kg: KgKey,
        neighbor_size: int = 20,
        embedding_name: str = "labse",
    ) -> tuple[list[int], dict[int, list[int]]]:
        cache_key = (kg, neighbor_size, embedding_name)
        if cache_key in self._neighbor_id_cache:
            return self._neighbor_id_cache[cache_key]

        embeddings = self.load_raw_embeddings(kg, embedding_name=embedding_name)
        ordered_ids = sorted(embeddings.keys())
        neighbor_ids: dict[int, list[int]] = {entity_id: [entity_id] for entity_id in ordered_ids}
        seen_neighbors: dict[int, set[int]] = {entity_id: {entity_id} for entity_id in ordered_ids}

        for triple in self.triples[kg]:
            head_id = triple.head_id
            tail_id = triple.tail_id
            if head_id in embeddings and tail_id in embeddings and tail_id not in seen_neighbors[head_id]:
                neighbor_ids[head_id].append(tail_id)
                seen_neighbors[head_id].add(tail_id)
            if head_id in embeddings and tail_id in embeddings and head_id not in seen_neighbors[tail_id]:
                neighbor_ids[tail_id].append(head_id)
                seen_neighbors[tail_id].add(head_id)

        trimmed = {entity_id: ids[:neighbor_size] for entity_id, ids in neighbor_ids.items()}
        self._neighbor_id_cache[cache_key] = (ordered_ids, trimmed)
        return ordered_ids, trimmed
