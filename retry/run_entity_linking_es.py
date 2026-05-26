#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

RETRY_DIR = Path(__file__).resolve().parent
if str(RETRY_DIR) not in sys.path:
    sys.path.insert(0, str(RETRY_DIR))

from entity_linking.config import DEFAULT_OUTPUT_DIR, REPO_ROOT
from entity_linking.es_eval import DEFAULT_FIND_FILE, evaluate_text_only, evaluate_vector_only, write_metrics_json
from entity_linking.es_index import DEFAULT_ES_INDEX_NAME, DEFAULT_ES_URL, index_processed_files


DEFAULT_EVAL_OUTPUT_DIR = RETRY_DIR / "output" / "entity_linking_eval"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Elasticsearch import and evaluation for recovered entity-linking data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Import processed entity-linking JSONL files into Elasticsearch")
    index_parser.add_argument("--input-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    index_parser.add_argument("--zh-input", type=Path, default=None)
    index_parser.add_argument("--en-input", type=Path, default=None)
    index_parser.add_argument("--index-name", default=DEFAULT_ES_INDEX_NAME)
    index_parser.add_argument("--es-url", default=DEFAULT_ES_URL)
    index_parser.add_argument("--batch-size", type=int, default=100)
    index_parser.add_argument("--vector-dim", type=int, default=1024)
    index_parser.add_argument("--no-recreate", action="store_true")
    index_parser.add_argument("--json", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="Evaluate text-only / vector-only retrieval against find.xlsx")
    eval_parser.add_argument("--mode", choices=("text", "vector", "both"), default="both")
    eval_parser.add_argument("--find-file", type=Path, default=DEFAULT_FIND_FILE)
    eval_parser.add_argument(
        "--vector-model-dir",
        type=Path,
        default=None,
        help="Fine-tuned entity-linking model directory used for vector encoding",
    )
    eval_parser.add_argument("--index-name", default=DEFAULT_ES_INDEX_NAME)
    eval_parser.add_argument("--es-url", default=DEFAULT_ES_URL)
    eval_parser.add_argument("--output-dir", type=Path, default=DEFAULT_EVAL_OUTPUT_DIR)
    eval_parser.add_argument("--json", action="store_true")

    return parser


def _resolved_inputs(args) -> tuple[Path, Path]:
    zh_input = args.zh_input or (args.input_dir / "entity_words_zh.jsonl")
    en_input = args.en_input or (args.input_dir / "entity_words_en.jsonl")
    return zh_input, en_input


def handle_index(args) -> int:
    zh_input, en_input = _resolved_inputs(args)
    summary = index_processed_files(
        zh_input=zh_input,
        en_input=en_input,
        index_name=args.index_name,
        es_url=args.es_url,
        batch_size=args.batch_size,
        vector_dim=args.vector_dim,
        recreate=not args.no_recreate,
    )
    payload = summary.to_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"ES index import completed: {payload['index_name']}")
    print(f"  indexed_documents: {payload['indexed_documents']}")
    print(f"  skipped_documents: {payload['skipped_documents']}")
    print(f"  documents_with_vectors: {payload['documents_with_vectors']}")
    print(f"  final_document_count: {payload['final_document_count']}")
    return 0


def handle_eval(args) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, object]] = {}

    if args.mode in {"text", "both"}:
        text_metrics = evaluate_text_only(
            index_name=args.index_name,
            es_url=args.es_url,
            find_path=args.find_file,
        )
        write_metrics_json(args.output_dir / "text_only_metrics.json", text_metrics)
        results["text_only"] = text_metrics.to_dict()

    if args.mode in {"vector", "both"}:
        if args.vector_model_dir is None:
            raise ValueError("--vector-model-dir is required for vector/both evaluation and must point to the fine-tuned entity-linking model directory")
        vector_metrics = evaluate_vector_only(
            vector_model_dir=args.vector_model_dir,
            index_name=args.index_name,
            es_url=args.es_url,
            find_path=args.find_file,
        )
        write_metrics_json(args.output_dir / "vector_only_metrics.json", vector_metrics)
        results["vector_only"] = vector_metrics.to_dict()

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    for name, metrics in results.items():
        print(name)
        print(f"  MRR: {metrics['mrr']:.4f}")
        print(f"  Hits@1: {metrics['hits@1']:.4f}")
        print(f"  Hits@5: {metrics['hits@5']:.4f}")
        print(f"  Hits@10: {metrics['hits@10']:.4f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "index":
        return handle_index(args)
    if args.command == "eval":
        return handle_eval(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
