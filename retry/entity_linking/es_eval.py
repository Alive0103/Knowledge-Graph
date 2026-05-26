from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence
from urllib.parse import unquote, urlsplit, urlunsplit

import pandas as pd

from .backends import TransformerVectorizer
from .config import REPO_ROOT
from .es_index import DEFAULT_ES_INDEX_NAME, DEFAULT_ES_URL, create_es_client


logger = logging.getLogger(__name__)

DEFAULT_FIND_FILE = REPO_ROOT / "work_wyy" / "data" / "find.xlsx"


@dataclass(frozen=True)
class RetrievalMetrics:
    mode: str
    index_name: str
    query_count: int
    mrr: float
    hits_at_1: float
    hits_at_5: float
    hits_at_10: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["hits@1"] = payload.pop("hits_at_1")
        payload["hits@5"] = payload.pop("hits_at_5")
        payload["hits@10"] = payload.pop("hits_at_10")
        return payload


def read_find_pairs(find_path: Path = DEFAULT_FIND_FILE) -> list[tuple[str, str]]:
    if not find_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {find_path}")
    frame = pd.read_excel(find_path, header=None, engine="openpyxl")
    pairs: list[tuple[str, str]] = []
    for row in frame.itertuples(index=False):
        if len(row) < 2:
            continue
        query = normalize_query(row[0])
        correct_link = str(row[1]).strip()
        if not query or not correct_link:
            continue
        pairs.append((query, correct_link))
    return pairs


def normalize_query(query: object) -> str:
    text = str(query or "")
    text = text.replace("\r", " ").replace("\n", " ")
    return " ".join(text.split())


def text_search(es, index_name: str, query: str, limit: int = 10) -> list[dict[str, object]]:
    body = {
        "query": {
            "bool": {
                "should": [
                    {"match": {"label": {"query": query, "boost": 4.0}}},
                    {"match": {"aliases_zh": {"query": query, "boost": 3.0}}},
                    {"match": {"aliases_en": {"query": query, "boost": 3.0}}},
                    {"match": {"descriptions_zh": {"query": query, "boost": 1.5}}},
                    {"match": {"descriptions_en": {"query": query, "boost": 1.5}}},
                ],
                "minimum_should_match": 1,
            }
        },
        "size": max(limit * 3, 20),
    }
    response = es.search(index=index_name, body=body)
    hits = response.get("hits", {}).get("hits", [])
    results: list[dict[str, object]] = []
    seen_links: set[str] = set()
    for hit in hits:
        source = hit.get("_source", {})
        link = str(source.get("link") or "").strip()
        key = link or str(hit.get("_id") or "")
        if not key or key in seen_links:
            continue
        seen_links.add(key)
        results.append(
            {
                "label": str(source.get("label") or ""),
                "link": link,
                "score": float(hit.get("_score") or 0.0),
            }
        )
        if len(results) >= limit:
            break
    return results


def vector_search(
    es,
    index_name: str,
    query: str,
    vectorizer: TransformerVectorizer,
    query_vector: list[float] | None = None,
    limit: int = 10,
) -> list[dict[str, object]]:
    if query_vector is None:
        query_vector = vectorizer.vectorize_terms([query])
    if query_vector is None:
        return []

    candidate_limit = max(limit * 2, 20)
    fields = (
        "entity_words_zh_vector",
        "entity_words_en_vector",
    )
    merged: dict[str, dict[str, object]] = {}

    for field_name in fields:
        body = {
            "knn": {
                "field": field_name,
                "query_vector": query_vector,
                "k": candidate_limit,
                "num_candidates": max(candidate_limit * 2, 50),
            },
            "size": candidate_limit,
        }
        try:
            response = es.search(index=index_name, body=body)
        except Exception as exc:
            logger.debug("Skipping vector field %s because ES query failed: %s", field_name, exc)
            continue

        hits = response.get("hits", {}).get("hits", [])
        for hit in hits:
            source = hit.get("_source", {})
            link = str(source.get("link") or "").strip()
            key = link or str(hit.get("_id") or "")
            if not key:
                continue
            score = float(hit.get("_score") or 0.0)
            current = merged.get(key)
            if current is None or score > float(current["score"]):
                merged[key] = {
                    "label": str(source.get("label") or ""),
                    "link": link,
                    "score": score,
                    "field": field_name,
                }

    ranked = sorted(merged.values(), key=lambda item: float(item["score"]), reverse=True)
    return ranked[:limit]


def _rank_for_link(results: Sequence[dict[str, object]], correct_link: str) -> int | None:
    expected = normalize_link(correct_link)
    for index, result in enumerate(results, start=1):
        candidate_link = normalize_link(result.get("link"))
        if expected and candidate_link == expected:
            return index
    return None


def normalize_link(link: object) -> str:
    text = str(link or "").strip()
    if not text:
        return ""
    parsed = urlsplit(text)
    if parsed.scheme and parsed.netloc:
        path = unquote(parsed.path).rstrip("/")
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, "", ""))
    return unquote(text).rstrip("/")


def _metrics_from_ranks(mode: str, index_name: str, ranks: list[int | None]) -> RetrievalMetrics:
    query_count = len(ranks)
    reciprocal_rank_sum = 0.0
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0

    for rank in ranks:
        if rank is None:
            continue
        reciprocal_rank_sum += 1.0 / rank
        if rank <= 1:
            hits_at_1 += 1
        if rank <= 5:
            hits_at_5 += 1
        if rank <= 10:
            hits_at_10 += 1

    total = max(query_count, 1)
    return RetrievalMetrics(
        mode=mode,
        index_name=index_name,
        query_count=query_count,
        mrr=round(reciprocal_rank_sum / total, 4),
        hits_at_1=round(hits_at_1 / total, 4),
        hits_at_5=round(hits_at_5 / total, 4),
        hits_at_10=round(hits_at_10 / total, 4),
    )


def evaluate_text_only(
    index_name: str = DEFAULT_ES_INDEX_NAME,
    es_url: str = DEFAULT_ES_URL,
    find_path: Path = DEFAULT_FIND_FILE,
) -> RetrievalMetrics:
    pairs = read_find_pairs(find_path=find_path)
    es = create_es_client(es_url=es_url)
    ranks: list[int | None] = []
    for query, correct_link in pairs:
        results = text_search(es=es, index_name=index_name, query=query, limit=10)
        ranks.append(_rank_for_link(results, correct_link))
    return _metrics_from_ranks("text_only", index_name=index_name, ranks=ranks)


def evaluate_vector_only(
    vector_model_dir: Path,
    index_name: str = DEFAULT_ES_INDEX_NAME,
    es_url: str = DEFAULT_ES_URL,
    find_path: Path = DEFAULT_FIND_FILE,
) -> RetrievalMetrics:
    pairs = read_find_pairs(find_path=find_path)
    es = create_es_client(es_url=es_url)
    vectorizer = TransformerVectorizer(str(vector_model_dir), dim=1024, batch_size=32)
    query_vectors = batch_encode_queries(vectorizer, [query for query, _ in pairs])
    ranks: list[int | None] = []
    for (query, correct_link), query_vector in zip(pairs, query_vectors):
        results = vector_search(
            es=es,
            index_name=index_name,
            query=query,
            vectorizer=vectorizer,
            query_vector=query_vector,
            limit=10,
        )
        ranks.append(_rank_for_link(results, correct_link))
    return _metrics_from_ranks("vector_only", index_name=index_name, ranks=ranks)


def write_metrics_json(path: Path, metrics: RetrievalMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics.to_dict(), handle, ensure_ascii=False, indent=2)


def batch_encode_queries(vectorizer: TransformerVectorizer, queries: Sequence[str]) -> list[list[float] | None]:
    results: list[list[float] | None] = []
    torch = vectorizer.torch

    with torch.no_grad():
        for start in range(0, len(queries), vectorizer.batch_size):
            batch_queries = [normalize_query(query) for query in queries[start:start + vectorizer.batch_size]]
            if not batch_queries:
                continue
            encoded = vectorizer.tokenizer(
                batch_queries,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            encoded = {key: value.to(vectorizer.device) for key, value in encoded.items()}
            outputs = vectorizer.model(**encoded)
            batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            for vector in batch_vectors:
                resized = vectorizer._resize(vector)
                norm = float((resized ** 2).sum() ** 0.5)
                if norm <= 0:
                    results.append(None)
                else:
                    results.append((resized / norm).astype("float32").tolist())

    return results
