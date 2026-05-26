from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from elasticsearch import Elasticsearch, helpers

from .io_utils import iter_jsonl


logger = logging.getLogger(__name__)

DEFAULT_ES_URL = os.getenv("KG_ES_URL", "http://localhost:9200")
DEFAULT_ES_INDEX_NAME = os.getenv("KG_ES_INDEX_NAME", "data2")
DEFAULT_REQUEST_TIMEOUT = int(os.getenv("KG_ES_REQUEST_TIMEOUT", "60"))


@dataclass(frozen=True)
class ESIndexSummary:
    index_name: str
    es_url: str
    zh_input: str
    en_input: str
    indexed_documents: int
    skipped_documents: int
    documents_with_vectors: int
    final_document_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def create_es_client(
    es_url: str = DEFAULT_ES_URL,
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
) -> Elasticsearch:
    compat_mode = "aliyuncs.com" in es_url
    kwargs: dict[str, object] = {
        "request_timeout": request_timeout,
    }
    if compat_mode:
        kwargs["headers"] = {
            "accept": "application/vnd.elasticsearch+json;compatible-with=8",
            "content-type": "application/vnd.elasticsearch+json;compatible-with=8",
        }
        kwargs["http_compress"] = True
    return Elasticsearch(es_url, **kwargs)


def build_index_mapping(vector_dim: int = 1024) -> dict[str, object]:
    dense_vector = {
        "type": "dense_vector",
        "dims": vector_dim,
        "index": True,
        "similarity": "cosine",
    }
    return {
        "mappings": {
            "properties": {
                "label": {"type": "text"},
                "link": {"type": "keyword"},
                "aliases_zh": {"type": "text"},
                "aliases_en": {"type": "text"},
                "descriptions_zh": {"type": "text"},
                "descriptions_en": {"type": "text"},
                "content": {"type": "text"},
                "entity_words_zh": {"type": "text"},
                "entity_words_en": {"type": "text"},
                "entity_words_zh_vector": dense_vector,
                "entity_words_en_vector": dense_vector,
                "label_vector": dense_vector,
                "label_zh_vector": dense_vector,
                "label_en_vector": dense_vector,
                "descriptions_zh_vector": dense_vector,
                "descriptions_en_vector": dense_vector,
            }
        }
    }


def recreate_index(client: Elasticsearch, index_name: str, vector_dim: int = 1024) -> None:
    if client.indices.exists(index=index_name):
        logger.info("Deleting existing ES index: %s", index_name)
        client.indices.delete(index=index_name)
    logger.info("Creating ES index: %s", index_name)
    mapping = build_index_mapping(vector_dim=vector_dim)
    try:
        client.indices.create(index=index_name, body=mapping)
    except TypeError:
        client.indices.create(index=index_name, mappings=mapping["mappings"])


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        cleaned = [_normalize_text(item) for item in value]
        return [item for item in cleaned if item]
    text = _normalize_text(value)
    return [text] if text else []


def _coerce_vector(value: object) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        vector = value.astype(np.float32).reshape(-1).tolist()
    else:
        vector = list(value)
    if not vector:
        return None
    return [float(item) for item in vector]


def _merge_vectors(*vectors: list[float] | None) -> list[float] | None:
    valid = [np.asarray(vector, dtype=np.float32) for vector in vectors if vector]
    if not valid:
        return None
    merged = np.mean(np.stack(valid, axis=0), axis=0)
    norm = float(np.linalg.norm(merged))
    if norm <= 0:
        return None
    return (merged / norm).astype(np.float32).tolist()


def _document_id(record: dict) -> str:
    stable_key = _normalize_text(record.get("wikipediaLink")) or _normalize_text(record.get("label"))
    digest = hashlib.sha1(stable_key.encode("utf-8")).hexdigest()
    return digest


def transform_record(record: dict) -> dict[str, object] | None:
    label = _normalize_text(record.get("label"))
    link = _normalize_text(record.get("wikipediaLink") or record.get("link"))
    aliases_zh = _normalize_list(record.get("zh_aliases") or record.get("aliases_zh"))
    aliases_en = _normalize_list(record.get("en_aliases") or record.get("aliases_en"))
    descriptions_zh = _normalize_text(record.get("zh_description") or record.get("descriptions_zh"))
    descriptions_en = _normalize_text(record.get("en_description") or record.get("descriptions_en"))
    content = _normalize_text(record.get("content"))
    entity_words_zh = _normalize_list(record.get("_entity_words_zh") or record.get("entity_words_zh"))
    entity_words_en = _normalize_list(record.get("_entity_words_en") or record.get("entity_words_en"))

    if not label and not link:
        return None

    zh_vector = _coerce_vector(record.get("_entity_words_zh_vector") or record.get("entity_words_zh_vector"))
    en_vector = _coerce_vector(record.get("_entity_words_en_vector") or record.get("entity_words_en_vector"))
    merged_vector = _merge_vectors(zh_vector, en_vector)

    payload: dict[str, object] = {
        "label": label,
        "link": link,
        "aliases_zh": aliases_zh,
        "aliases_en": aliases_en,
        "descriptions_zh": descriptions_zh,
        "descriptions_en": descriptions_en,
        "content": content,
        "entity_words_zh": entity_words_zh,
        "entity_words_en": entity_words_en,
    }

    if zh_vector is not None:
        payload["entity_words_zh_vector"] = zh_vector
        payload["label_zh_vector"] = zh_vector
        payload["descriptions_zh_vector"] = zh_vector
    if en_vector is not None:
        payload["entity_words_en_vector"] = en_vector
        payload["label_en_vector"] = en_vector
        payload["descriptions_en_vector"] = en_vector
    if merged_vector is not None:
        payload["label_vector"] = merged_vector

    return payload


def index_processed_files(
    zh_input: Path,
    en_input: Path,
    index_name: str = DEFAULT_ES_INDEX_NAME,
    es_url: str = DEFAULT_ES_URL,
    batch_size: int = 100,
    vector_dim: int = 1024,
    recreate: bool = True,
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
) -> ESIndexSummary:
    client = create_es_client(es_url=es_url, request_timeout=request_timeout)
    if recreate:
        recreate_index(client, index_name=index_name, vector_dim=vector_dim)

    indexed_documents = 0
    skipped_documents = 0
    documents_with_vectors = 0
    actions: list[dict[str, object]] = []

    def flush_batch() -> None:
        nonlocal indexed_documents, skipped_documents
        if not actions:
            return
        success_count, errors = helpers.bulk(
            client,
            actions,
            stats_only=False,
            raise_on_error=False,
            request_timeout=request_timeout,
        )
        indexed_documents += int(success_count)
        skipped_documents += len(errors)
        actions.clear()

    for path in (zh_input, en_input):
        if not path.exists():
            raise FileNotFoundError(f"Processed entity-linking file not found: {path}")
        logger.info("Indexing processed entity-linking file: %s", path)
        for record in iter_jsonl(path):
            payload = transform_record(record)
            if payload is None:
                skipped_documents += 1
                continue
            if any(
                payload.get(key) is not None
                for key in ("entity_words_zh_vector", "entity_words_en_vector", "label_vector")
            ):
                documents_with_vectors += 1
            actions.append(
                {
                    "_index": index_name,
                    "_id": _document_id(record),
                    "_source": payload,
                }
            )
            if len(actions) >= batch_size:
                flush_batch()

    flush_batch()
    client.indices.refresh(index=index_name)
    final_document_count = int(client.count(index=index_name)["count"])

    return ESIndexSummary(
        index_name=index_name,
        es_url=es_url,
        zh_input=str(zh_input),
        en_input=str(en_input),
        indexed_documents=indexed_documents,
        skipped_documents=skipped_documents,
        documents_with_vectors=documents_with_vectors,
        final_document_count=final_document_count,
    )
