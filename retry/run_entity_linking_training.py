#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

RETRY_DIR = Path(__file__).resolve().parent
if str(RETRY_DIR) not in sys.path:
    sys.path.insert(0, str(RETRY_DIR))

from entity_linking.download import DEFAULT_BASE_MODEL_DIR, DEFAULT_BASE_MODEL_NAME, download_base_model
from entity_linking.training import NERTrainingConfig, train_token_classifier
from entity_linking.training_data import (
    DEFAULT_SUPERVISED_TRAINDATA_DIR,
    DEFAULT_TRAINING_DATA_DIR,
    SupervisedNERDataConfig,
    WeakNERDataConfig,
    build_supervised_ner_dataset,
    build_weak_ner_dataset,
    has_supervised_traindata,
)


DEFAULT_FINETUNED_DIR = DEFAULT_TRAINING_DATA_DIR / "ner_finetuned"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the entity-linking NER model with supervised traindata or weak supervision")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL_NAME, help="Hugging Face base model name")
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL_DIR, help="Local directory for the downloaded base model")
    parser.add_argument("--hf-endpoint", default=None, help="Optional Hugging Face endpoint or mirror")
    parser.add_argument("--training-data-dir", type=Path, default=DEFAULT_TRAINING_DATA_DIR, help="Directory for normalized NER train/dev files")
    parser.add_argument("--training-data-source", choices=("auto", "supervised", "weak"), default="auto", help="Prefer supervised traindata when available, otherwise fall back to weak supervision")
    parser.add_argument("--supervised-traindata-dir", type=Path, default=DEFAULT_SUPERVISED_TRAINDATA_DIR, help="Directory of work_wyy traindata files such as *_ner_train.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FINETUNED_DIR, help="Directory for the fine-tuned NER model")
    parser.add_argument("--max-records", type=int, default=None, help="Optional record limit for weak-supervision dataset generation")
    parser.add_argument("--max-train-examples", type=int, default=None, help="Optional limit for normalized supervised train examples")
    parser.add_argument("--max-dev-examples", type=int, default=None, help="Optional limit for normalized supervised dev examples")
    parser.add_argument("--max-test-examples", type=int, default=None, help="Optional limit for normalized supervised test examples")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")

    data_source = args.training_data_source
    training_data_payload: dict[str, object]
    supervised_available = has_supervised_traindata(args.supervised_traindata_dir)

    if data_source == "supervised" or (data_source == "auto" and supervised_available):
        if not supervised_available:
            raise FileNotFoundError(f"Supervised traindata not found: {args.supervised_traindata_dir}")
        data_config = SupervisedNERDataConfig(
            source_dir=args.supervised_traindata_dir,
            output_dir=args.training_data_dir,
            max_train_examples=args.max_train_examples,
            max_dev_examples=args.max_dev_examples,
            max_test_examples=args.max_test_examples,
        )
        train_path, dev_path, test_path, data_stats = build_supervised_ner_dataset(data_config)
        training_data_payload = {
            "source": "supervised_traindata",
            "source_dir": str(args.supervised_traindata_dir),
            "train_path": str(train_path),
            "dev_path": str(dev_path),
            "test_path": str(test_path),
            "stats": data_stats.__dict__,
        }
    else:
        data_config = WeakNERDataConfig(
            output_dir=args.training_data_dir,
            max_records=args.max_records,
        )
        train_path, dev_path, data_stats = build_weak_ner_dataset(data_config)
        training_data_payload = {
            "source": "weak_supervision",
            "train_path": str(train_path),
            "dev_path": str(dev_path),
            "stats": data_stats.__dict__,
        }

    if args.skip_download:
        model_dir = args.base_model_dir
    else:
        model_dir = download_base_model(
            model_name=args.base_model,
            output_dir=args.base_model_dir,
            overwrite=args.overwrite,
            hf_endpoint=args.hf_endpoint,
        )

    training_config = NERTrainingConfig(
        model_name_or_path=str(model_dir),
        train_path=train_path,
        dev_path=dev_path,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        max_train_examples=None,
        max_dev_examples=None,
    )
    model_output_dir, training_summary = train_token_classifier(training_config)

    payload = {
        "training_data": training_data_payload,
        "base_model_dir": str(model_dir),
        "finetuned_model_dir": str(model_output_dir),
        "training_summary": training_summary.__dict__,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
