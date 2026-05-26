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

from alignment.config import DEFAULT_DATASET, DEFAULT_DBP15K_ROOT
from alignment.dbp15k import DBP15KDataset
from alignment.embedding_builder import (
    DEFAULT_BGE_M3_EMBEDDING_PREFIX,
    DEFAULT_BGE_M3_EMBEDDING_KEY,
    DEFAULT_BGE_M3_MODEL_DIR,
    DEFAULT_BGE_M3_MODEL_NAME,
    build_name_embedding_pickles,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build alternative raw embedding baselines for DBP15K entity alignment")
    parser.add_argument("--dbp15k-root", type=Path, default=DEFAULT_DBP15K_ROOT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--model-name", default=DEFAULT_BGE_M3_MODEL_NAME)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_BGE_M3_MODEL_DIR)
    parser.add_argument("--embedding-name", default=DEFAULT_BGE_M3_EMBEDDING_KEY)
    parser.add_argument("--output-prefix", default=DEFAULT_BGE_M3_EMBEDDING_PREFIX)
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset = DBP15KDataset(dataset_dir=Path(args.dbp15k_root) / args.dataset, dataset_name=args.dataset)
    summary = build_name_embedding_pickles(
        dataset=dataset,
        model_name=args.model_name,
        model_dir=args.model_dir,
        embedding_name=args.embedding_name,
        output_prefix=args.output_prefix,
        overwrite=args.overwrite,
        hf_endpoint=args.hf_endpoint,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
