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

from alignment.config import DEFAULT_DATASET, DEFAULT_MODEL_PATH, REPO_ROOT
from alignment.embedding_builder import DEFAULT_BGE_M3_MODEL_DIR, DEFAULT_BGE_M3_MODEL_NAME
from alignment.prepare import prepare_alignment_dataset
from entity_linking.download import DEFAULT_BASE_MODEL_DIR, DEFAULT_BASE_MODEL_NAME, download_base_model
from entity_linking.es_index import DEFAULT_ES_URL, create_es_client
from model_hub import download_hf_snapshot, looks_like_model_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare local data and model assets for the full retry experiment")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL_NAME)
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL_DIR)
    parser.add_argument("--final-model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prepare-bge-model", action="store_true")
    parser.add_argument("--bge-model-name", default=DEFAULT_BGE_M3_MODEL_NAME)
    parser.add_argument("--bge-model-dir", type=Path, default=DEFAULT_BGE_M3_MODEL_DIR)
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument("--check-es", action="store_true")
    parser.add_argument("--es-url", default=DEFAULT_ES_URL)
    parser.add_argument("--json", action="store_true")
    return parser


def _verify_path(path: Path | None, label: str) -> dict[str, object]:
    return {
        "label": label,
        "path": None if path is None else str(path),
        "exists": False if path is None else path.exists(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    alignment_summary = prepare_alignment_dataset(dataset=args.dataset, repo_root=REPO_ROOT)
    if not alignment_summary.is_complete:
        raise RuntimeError(f"Alignment dataset recovery failed for {args.dataset}: {alignment_summary.target_dataset_dir}")

    base_model_dir = download_base_model(
        model_name=args.base_model,
        output_dir=args.base_model_dir,
        hf_endpoint=args.hf_endpoint,
    )

    bge_model_dir = Path(args.bge_model_dir)
    if args.prepare_bge_model:
        bge_model_dir = download_hf_snapshot(
            model_name=args.bge_model_name,
            output_dir=bge_model_dir,
            hf_endpoint=args.hf_endpoint,
        )
    bge_model_ready = looks_like_model_dir(Path(bge_model_dir))

    es_info = None
    if args.check_es:
        es = create_es_client(es_url=args.es_url)
        es_info = dict(es.info())

    required_paths = [
        _verify_path(REPO_ROOT / "work_wyy" / "data" / "zh_wiki_v2.jsonl", "zh wiki raw"),
        _verify_path(REPO_ROOT / "work_wyy" / "data" / "en_wiki_v3.jsonl", "en wiki raw"),
        _verify_path(REPO_ROOT / "work_wyy" / "data" / "find.xlsx", "entity linking eval xlsx"),
        _verify_path(Path(base_model_dir), "entity linking base model"),
        _verify_path(args.final_model_path, "alignment final model"),
    ]

    payload = {
        "repo_root": str(REPO_ROOT),
        "alignment": alignment_summary.to_dict(),
        "base_model_dir": str(base_model_dir),
        "bge_model_dir": str(bge_model_dir),
        "bge_model_ready": bge_model_ready,
        "required_paths": required_paths,
        "es_info": es_info,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
