#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

RETRY_DIR = Path(__file__).resolve().parent
if str(RETRY_DIR) not in sys.path:
    sys.path.insert(0, str(RETRY_DIR))

from alignment.config import DEFAULT_DATASET, DEFAULT_DBP15K_ROOT
from alignment.prepare import prepare_alignment_dataset
from alignment.training import (
    DEFAULT_ALIGNMENT_TRAINING_OUTPUT_DIR,
    DEFAULT_LABSE_MODEL_DIR,
    DEFAULT_LABSE_MODEL_NAME,
    AlignmentTrainingConfig,
    train_alignment_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the DBP15K neighbor-graph entity alignment model from scratch"
    )
    parser.add_argument("--dbp15k-root", type=Path, default=DEFAULT_DBP15K_ROOT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embedding-name", choices=("labse", "bge_m3"), default="labse")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--queue-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--momentum", type=float, default=0.9999)
    parser.add_argument("--neighbor-size", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--gat-num", type=int, default=1)
    parser.add_argument("--selection-metric", choices=("valid_hits@1", "valid_hits@10", "valid_mrr"), default="valid_hits@1")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--log-every-updates", type=int, default=50)
    parser.add_argument("--max-train-updates-per-epoch", type=int, default=None)
    parser.add_argument("--embedding-model-name", default=None)
    parser.add_argument("--embedding-model-dir", type=Path, default=None)
    parser.add_argument("--labse-model-name", default=DEFAULT_LABSE_MODEL_NAME)
    parser.add_argument("--labse-model-dir", type=Path, default=DEFAULT_LABSE_MODEL_DIR)
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument("--embedding-build-batch-size", type=int, default=32)
    parser.add_argument("--embedding-build-max-length", type=int, default=96)
    parser.add_argument("--json", action="store_true")
    return parser


def _default_output_dir(dataset: str, run_tag: str, embedding_name: str) -> Path:
    return DEFAULT_ALIGNMENT_TRAINING_OUTPUT_DIR / f"{embedding_name}_neighbor_retrained_{dataset}_{run_tag}"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    stamp = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or _default_output_dir(args.dataset, stamp, args.embedding_name)

    prepared = prepare_alignment_dataset(dataset=args.dataset)
    dataset_dir = Path(args.dbp15k_root) / args.dataset
    if not dataset_dir.exists() or not (dataset_dir / "triples_1").exists():
        dataset_dir = Path(prepared.target_dataset_dir)

    config = AlignmentTrainingConfig(
        dataset_dir=dataset_dir,
        output_dir=Path(output_dir),
        dataset_name=args.dataset,
        device=args.device,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        queue_length=args.queue_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        temperature=args.temperature,
        momentum=args.momentum,
        neighbor_size=args.neighbor_size,
        dropout=args.dropout,
        gat_num=args.gat_num,
        selection_metric=args.selection_metric,
        seed=args.seed,
        log_every_updates=args.log_every_updates,
        max_train_updates_per_epoch=args.max_train_updates_per_epoch,
        embedding_name=args.embedding_name,
        embedding_model_name=args.embedding_model_name,
        embedding_model_dir=args.embedding_model_dir,
        labse_model_name=args.labse_model_name,
        labse_model_dir=args.labse_model_dir,
        hf_endpoint=args.hf_endpoint,
        embedding_build_batch_size=args.embedding_build_batch_size,
        embedding_build_max_length=args.embedding_build_max_length,
    )

    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    summary = train_alignment_model(config)
    payload = {
        "prepared_dataset": prepared.to_dict(),
        "training_summary": summary.to_dict(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
