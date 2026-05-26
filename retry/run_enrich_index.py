#!/usr/bin/env python3
"""
Build an enriched Elasticsearch index by merging aligned English entity content
into Chinese entity records, then indexing everything into a new ES index.

Steps:
  1. Load alignment predictions (from run_predict_alignments.py)
  2. Load entity_words_zh.jsonl and entity_words_en.jsonl
  3. For each zh entity with a predicted en alignment:
       - Merge English aliases, description, entity words, and vectors
  4. Collect unmatched English entities as standalone documents
  5. Build new ES index via the existing index_processed_files()

Usage:
  python retry/run_enrich_index.py [--index-name data2_enriched_aligned] [--score-threshold 0.0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

RETRY_DIR = Path(__file__).resolve().parent
if str(RETRY_DIR) not in sys.path:
    sys.path.insert(0, str(RETRY_DIR))

from entity_linking.config import DEFAULT_OUTPUT_DIR
from entity_linking.es_index import DEFAULT_ES_INDEX_NAME, DEFAULT_ES_URL, index_processed_files
from entity_linking.io_utils import iter_jsonl

PREDICTIONS_FILE = RETRY_DIR / "output" / "alignment_predictions" / "bge_m3_graph_predictions.json"
DEFAULT_ZH_INPUT = DEFAULT_OUTPUT_DIR / "entity_words_zh.jsonl"
DEFAULT_EN_INPUT = DEFAULT_OUTPUT_DIR / "entity_words_en.jsonl"
DEFAULT_ENRICHED_INDEX = "data2_enriched_aligned"


def _normalize_list(value: object) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _merge_vectors(v1: list[float] | None, v2: list[float] | None) -> list[float] | None:
    if v1 is None and v2 is None:
        return None
    if v1 is None:
        return v2
    if v2 is None:
        return v1
    a1, a2 = np.asarray(v1, dtype=np.float32), np.asarray(v2, dtype=np.float32)
    merged = (a1 + a2) / 2.0
    norm = float(np.linalg.norm(merged))
    if norm <= 0:
        return v1
    return (merged / norm).tolist()


def _get_vector(doc: dict, key_prefixed: str, key_plain: str) -> list[float] | None:
    v = doc.get(key_prefixed) or doc.get(key_plain)
    if v is None:
        return None
    return list(v) if not isinstance(v, list) else v


def enrich_and_index(
    predictions_file: Path,
    zh_input: Path,
    en_input: Path,
    index_name: str,
    es_url: str,
    score_threshold: float,
    output_dir: Path,
    include_unmatched_en: bool = True,
) -> None:
    # --- 1. Load predictions ---
    print(f"Loading predictions from {predictions_file}")
    with open(predictions_file, "r", encoding="utf-8") as f:
        raw_predictions: list[dict] = json.load(f)

    # Filter by score threshold, build {zh_link → {en_link, score}}
    zh_to_en: dict[str, dict] = {}
    for p in raw_predictions:
        if p.get("score", 0.0) >= score_threshold and p.get("zh_link") and p.get("en_link"):
            zh_to_en[p["zh_link"]] = {"en_link": p["en_link"], "score": p["score"]}
    print(f"  {len(zh_to_en)} predictions above threshold {score_threshold}")

    # --- 2. Load en entities into dict by link ---
    print(f"Loading English entities from {en_input}")
    en_by_link: dict[str, dict] = {}
    for doc in iter_jsonl(en_input):
        link = str(doc.get("wikipediaLink") or doc.get("link") or "").strip()
        if link:
            en_by_link[link] = doc
    print(f"  loaded {len(en_by_link)} English entities")

    # --- 3. Process Chinese entities, merging aligned English content ---
    print(f"Processing Chinese entities from {zh_input}")
    matched_en_links: set[str] = set()
    enriched_zh_records: list[dict] = []
    enriched_count = 0

    for zh_doc in iter_jsonl(zh_input):
        zh_link = str(zh_doc.get("wikipediaLink") or zh_doc.get("link") or "").strip()
        pred = zh_to_en.get(zh_link)

        if pred:
            en_link = pred["en_link"]
            en_doc = en_by_link.get(en_link)
            if en_doc:
                # Merge text fields
                zh_aliases_en = _normalize_list(zh_doc.get("en_aliases") or zh_doc.get("aliases_en"))
                en_aliases_en = _normalize_list(en_doc.get("en_aliases") or en_doc.get("aliases_en"))
                en_aliases_zh = _normalize_list(en_doc.get("zh_aliases") or en_doc.get("aliases_zh"))
                merged_aliases_en = list(dict.fromkeys(zh_aliases_en + en_aliases_en + en_aliases_zh))

                zh_desc_en = str(zh_doc.get("en_description") or zh_doc.get("descriptions_en") or "").strip()
                en_desc_en = str(en_doc.get("en_description") or en_doc.get("descriptions_en") or "").strip()
                if not zh_desc_en and en_desc_en:
                    merged_desc_en = en_desc_en
                elif zh_desc_en and en_desc_en and en_desc_en not in zh_desc_en:
                    merged_desc_en = zh_desc_en + " " + en_desc_en
                else:
                    merged_desc_en = zh_desc_en or en_desc_en

                zh_words_en = _normalize_list(zh_doc.get("_entity_words_en") or zh_doc.get("entity_words_en"))
                en_words_en = _normalize_list(en_doc.get("_entity_words_en") or en_doc.get("entity_words_en"))
                merged_words_en = list(dict.fromkeys(zh_words_en + en_words_en))

                zh_vec_en = _get_vector(zh_doc, "_entity_words_en_vector", "entity_words_en_vector")
                en_vec_en = _get_vector(en_doc, "_entity_words_en_vector", "entity_words_en_vector")
                merged_vec_en = _merge_vectors(zh_vec_en, en_vec_en)

                zh_doc = dict(zh_doc)
                zh_doc["en_aliases"] = merged_aliases_en
                zh_doc["en_description"] = merged_desc_en
                zh_doc["_entity_words_en"] = merged_words_en
                if merged_vec_en is not None:
                    zh_doc["_entity_words_en_vector"] = merged_vec_en

                matched_en_links.add(en_link)
                enriched_count += 1

        enriched_zh_records.append(zh_doc)

    print(f"  {enriched_count} Chinese entities enriched with English content")

    # --- 4. Collect unmatched English entities ---
    unmatched_en_records = [doc for link, doc in en_by_link.items() if link not in matched_en_links]
    if include_unmatched_en:
        print(f"  {len(unmatched_en_records)} unmatched English entities will be added as standalone documents")
    else:
        print(f"  {len(unmatched_en_records)} unmatched English entities skipped (--no-unmatched-en)")

    # --- 5. Write intermediate JSONL files ---
    output_dir.mkdir(parents=True, exist_ok=True)
    zh_enriched_path = output_dir / "entity_words_zh_enriched.jsonl"
    en_unmatched_path = output_dir / "entity_words_en_unmatched.jsonl"

    with open(zh_enriched_path, "w", encoding="utf-8") as f:
        for doc in enriched_zh_records:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"Written {len(enriched_zh_records)} enriched zh records → {zh_enriched_path}")

    if include_unmatched_en:
        with open(en_unmatched_path, "w", encoding="utf-8") as f:
            for doc in unmatched_en_records:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"Written {len(unmatched_en_records)} unmatched en records → {en_unmatched_path}")
    else:
        # Write empty file so index_processed_files can still be called uniformly
        with open(en_unmatched_path, "w", encoding="utf-8") as f:
            pass
        print(f"Written 0 unmatched en records (skipped)")

    # --- 6. Build ES index ---
    print(f"\nBuilding ES index '{index_name}' ...")
    summary = index_processed_files(
        zh_input=zh_enriched_path,
        en_input=en_unmatched_path,
        index_name=index_name,
        es_url=es_url,
        batch_size=100,
        vector_dim=1024,
        recreate=True,
    )
    print(f"Index built successfully:")
    print(f"  indexed_documents:       {summary.indexed_documents}")
    print(f"  skipped_documents:       {summary.skipped_documents}")
    print(f"  documents_with_vectors:  {summary.documents_with_vectors}")
    print(f"  final_document_count:    {summary.final_document_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich ES index with aligned English entity content")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_FILE)
    parser.add_argument("--zh-input", type=Path, default=DEFAULT_ZH_INPUT)
    parser.add_argument("--en-input", type=Path, default=DEFAULT_EN_INPUT)
    parser.add_argument("--index-name", default=DEFAULT_ENRICHED_INDEX)
    parser.add_argument("--es-url", default=DEFAULT_ES_URL)
    parser.add_argument("--score-threshold", type=float, default=0.0,
                        help="Minimum alignment score to use (0.0 = use all predictions)")
    parser.add_argument("--no-unmatched-en", action="store_true",
                        help="Do not add unmatched English entities as standalone documents")
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_OUTPUT_DIR / "enriched")
    args = parser.parse_args()

    enrich_and_index(
        predictions_file=args.predictions,
        zh_input=args.zh_input,
        en_input=args.en_input,
        index_name=args.index_name,
        es_url=args.es_url,
        score_threshold=args.score_threshold,
        output_dir=args.output_dir,
        include_unmatched_en=not args.no_unmatched_en,
    )


if __name__ == "__main__":
    main()
