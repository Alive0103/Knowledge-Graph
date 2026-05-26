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

from alignment.config import DEFAULT_DATASET, DEFAULT_DBP15K_ROOT, DEFAULT_MODEL_PATH
from alignment.dbp15k import DBP15KDataset
from alignment.evaluation import evaluate_final_model_alignment, evaluate_raw_alignment
from entity_linking.config import EntityLinkingConfig
from entity_linking.download import DEFAULT_BASE_MODEL_DIR, DEFAULT_BASE_MODEL_NAME, download_base_model
from entity_linking.pipeline import run_pipeline
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the retry-side pipeline from raw wiki data to entity linking and DBP15K alignment smoke evaluation")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL_NAME)
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL_DIR)
    parser.add_argument("--hf-endpoint", default=None, help="Optional Hugging Face endpoint or mirror")
    parser.add_argument("--training-data-dir", type=Path, default=DEFAULT_TRAINING_DATA_DIR)
    parser.add_argument("--training-data-source", choices=("auto", "supervised", "weak"), default="auto")
    parser.add_argument("--supervised-traindata-dir", type=Path, default=DEFAULT_SUPERVISED_TRAINDATA_DIR)
    parser.add_argument("--finetuned-model-dir", type=Path, default=None)
    parser.add_argument("--entity-linking-output-dir", type=Path, default=None)
    parser.add_argument("--vectorizer", choices=("none", "hash", "transformer"), default=None)
    parser.add_argument("--dbp15k-root", type=Path, default=DEFAULT_DBP15K_ROOT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--alignment-model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def _resolve_runtime(mode: str) -> dict[str, int | None]:
    if mode == "smoke":
        return {
            "data_max_records": 200,
            "train_epochs": 1,
            "train_batch_size": 4,
            "train_max_length": 192,
            "max_train_examples": 64,
            "max_dev_examples": 16,
            "max_test_examples": 16,
            "entity_linking_max_records": 20,
            "eval_batch_size": 64,
        }
    return {
        "data_max_records": None,
        "train_epochs": 2,
        "train_batch_size": 4,
        "train_max_length": 256,
        "max_train_examples": None,
        "max_dev_examples": None,
        "max_test_examples": None,
        "entity_linking_max_records": None,
        "eval_batch_size": 128,
    }


def _default_finetuned_dir(training_data_dir: Path, mode: str) -> Path:
    suffix = "ner_finetuned_smoke" if mode == "smoke" else "ner_finetuned"
    return training_data_dir / suffix


def _default_entity_linking_output_dir(training_data_dir: Path, mode: str) -> Path:
    base_dir = training_data_dir.parent
    suffix = "entity_linking_smoke_transformer" if mode == "smoke" else "entity_linking_transformer"
    return base_dir / suffix


def _prepare_training_data(args, runtime: dict[str, int | None], training_data_dir: Path) -> tuple[Path, Path, dict[str, object]]:
    supervised_available = has_supervised_traindata(args.supervised_traindata_dir)
    use_supervised = args.training_data_source == "supervised" or (
        args.training_data_source == "auto" and supervised_available
    )

    if use_supervised:
        if not supervised_available:
            raise FileNotFoundError(f"Supervised traindata not found: {args.supervised_traindata_dir}")
        train_path, dev_path, test_path, data_stats = build_supervised_ner_dataset(
            SupervisedNERDataConfig(
                source_dir=args.supervised_traindata_dir,
                output_dir=training_data_dir,
                max_train_examples=runtime["max_train_examples"],
                max_dev_examples=runtime["max_dev_examples"],
                max_test_examples=runtime["max_test_examples"],
            )
        )
        return train_path, dev_path, {
            "source": "supervised_traindata",
            "source_dir": str(args.supervised_traindata_dir),
            "train_path": str(train_path),
            "dev_path": str(dev_path),
            "test_path": str(test_path),
            "stats": data_stats.__dict__,
        }

    train_path, dev_path, data_stats = build_weak_ner_dataset(
        WeakNERDataConfig(
            output_dir=training_data_dir,
            max_records=runtime["data_max_records"],
        )
    )
    return train_path, dev_path, {
        "source": "weak_supervision",
        "train_path": str(train_path),
        "dev_path": str(dev_path),
        "stats": data_stats.__dict__,
    }


def _choose_sample_entity_id(dataset: DBP15KDataset) -> int:
    pairs = dataset.get_alignment_pairs("test")
    if pairs:
        return pairs[0].left_id
    return next(iter(dataset.entities["1"].keys()))


def _build_relation_payload(dataset: DBP15KDataset, kg: str, relation_id: int, limit: int = 5) -> dict[str, object]:
    relation = dataset.get_relation(kg, relation_id)
    triples = dataset.search_triples_by_relation(kg, relation_id=relation_id, limit=limit)
    return {
        "kg": kg,
        "relation_id": relation_id,
        "name": relation.name if relation else "",
        "sample_triples": [dataset.render_triple(triple) for triple in triples],
    }


def _alignment_smoke(dataset: DBP15KDataset, alignment_model_path: Path | None, batch_size: int, device: str) -> dict[str, object]:
    entity_id = _choose_sample_entity_id(dataset)
    entity_payload = dataset.describe_entity("1", entity_id, relation_limit=5, triple_limit=5)

    relation_summary = entity_payload["relation_summary"]
    if relation_summary:
        relation_id = int(relation_summary[0]["relation_id"])
    else:
        relation_id = next(iter(dataset.relations["1"].keys()))

    relation_payload = _build_relation_payload(dataset, "1", relation_id=relation_id, limit=5)
    relation_name = str(relation_payload["name"] or "")
    relation_query = relation_name[: max(1, min(4, len(relation_name)))] if relation_name else str(relation_id)
    search_relations = [
        {"relation_id": record.relation_id, "name": record.name}
        for record in dataset.search_relations("1", relation_query, limit=5)
    ]
    retrieval = [
        dataset.render_triple(triple)
        for triple in dataset.search_triples_by_relation("1", relation_id=relation_id, entity_id=entity_id, limit=5)
    ]
    alignment_lookup = {
        split_name: {"kg": record.kg, "entity_id": record.entity_id, "name": record.name}
        for split_name, record in dataset.find_alignment("1", entity_id, split="all").items()
    }

    result = {
        "entity_query": entity_payload,
        "relation_query": relation_payload,
        "search_relations": search_relations,
        "retrieve_triples": retrieval,
        "alignment_lookup": alignment_lookup,
        "raw_eval": evaluate_raw_alignment(dataset=dataset, split="test", batch_size=batch_size).to_dict(),
    }

    if alignment_model_path and alignment_model_path.exists():
        result["final_model_eval"] = evaluate_final_model_alignment(
            dataset=dataset,
            model_path=alignment_model_path,
            split="test",
            batch_size=batch_size,
            device=device,
        ).to_dict()
    else:
        result["final_model_eval"] = None

    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")

    runtime = _resolve_runtime(args.mode)
    training_data_dir = Path(args.training_data_dir)
    finetuned_model_dir = args.finetuned_model_dir or _default_finetuned_dir(training_data_dir, args.mode)
    entity_linking_output_dir = args.entity_linking_output_dir or _default_entity_linking_output_dir(training_data_dir, args.mode)
    vectorizer = args.vectorizer or ("hash" if args.mode == "smoke" else "transformer")

    train_path, dev_path, training_data_payload = _prepare_training_data(args, runtime, training_data_dir)

    base_model_dir = download_base_model(
        model_name=args.base_model,
        output_dir=args.base_model_dir,
        overwrite=args.overwrite,
        hf_endpoint=args.hf_endpoint,
    )

    finetuned_model_dir, training_summary = train_token_classifier(
        NERTrainingConfig(
            model_name_or_path=str(base_model_dir),
            train_path=train_path,
            dev_path=dev_path,
            output_dir=Path(finetuned_model_dir),
            max_length=int(runtime["train_max_length"]),
            batch_size=int(runtime["train_batch_size"]),
            epochs=int(runtime["train_epochs"]),
            device=args.device,
            max_train_examples=runtime["max_train_examples"],
            max_dev_examples=runtime["max_dev_examples"],
        )
    )

    entity_linking_config = EntityLinkingConfig(
        output_dir=Path(entity_linking_output_dir),
        extractor="transformer",
        vectorizer=vectorizer,
        transformer_ner_model=str(finetuned_model_dir),
        transformer_vector_model=str(finetuned_model_dir) if vectorizer == "transformer" else None,
        max_records=runtime["entity_linking_max_records"],
        overwrite=args.overwrite,
    )
    zh_stats, en_stats = run_pipeline(entity_linking_config)

    alignment_dataset = DBP15KDataset(dataset_dir=Path(args.dbp15k_root) / args.dataset, dataset_name=args.dataset)
    alignment_summary = _alignment_smoke(
        dataset=alignment_dataset,
        alignment_model_path=args.alignment_model_path,
        batch_size=int(runtime["eval_batch_size"]),
        device=args.device,
    )

    summary = {
        "mode": args.mode,
        "base_model": {
            "name": args.base_model,
            "dir": str(base_model_dir),
        },
        "training_data": training_data_payload,
        "ner_training": training_summary.__dict__,
        "entity_linking": {
            "output_dir": str(entity_linking_config.output_dir),
            "extractor": entity_linking_config.extractor,
            "vectorizer": entity_linking_config.vectorizer,
            "transformer_ner_model": entity_linking_config.transformer_ner_model,
            "transformer_vector_model": entity_linking_config.transformer_vector_model,
            "zh_stats": zh_stats.__dict__,
            "en_stats": en_stats.__dict__,
        },
        "alignment": alignment_summary,
    }

    summary_path = training_data_dir / f"full_pipeline_summary_{args.mode}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps({"summary_path": str(summary_path), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
