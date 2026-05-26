from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.config import DEFAULT_DATASET, DEFAULT_DBP15K_ROOT, DEFAULT_MODEL_PATH
from alignment.dbp15k import DBP15KDataset
from alignment.evaluation import evaluate_final_model_alignment, evaluate_raw_alignment
from alignment.prepare import prepare_alignment_dataset
from alignment.training import (
    DEFAULT_ALIGNMENT_TRAINING_OUTPUT_DIR,
    DEFAULT_LABSE_MODEL_DIR,
    DEFAULT_LABSE_MODEL_NAME,
    AlignmentTrainingConfig,
    train_alignment_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DBP15K query and evaluation utilities")
    parser.add_argument("--dbp15k-root", type=Path, default=DEFAULT_DBP15K_ROOT, help="Root directory of DBP15K processed data")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset name under the DBP15K root, e.g. zh_en")

    subparsers = parser.add_subparsers(dest="command", required=True)

    entity_parser = subparsers.add_parser("entity", help="Query an entity by numeric id")
    entity_parser.add_argument("--kg", choices=("1", "2"), required=True)
    entity_parser.add_argument("--id", type=int, required=True, dest="entity_id")
    entity_parser.add_argument("--relation-limit", type=int, default=10)
    entity_parser.add_argument("--triple-limit", type=int, default=10)
    entity_parser.add_argument("--json", action="store_true")

    relation_parser = subparsers.add_parser("relation", help="Query a relation by numeric id")
    relation_parser.add_argument("--kg", choices=("1", "2"), required=True)
    relation_parser.add_argument("--id", type=int, required=True, dest="relation_id")
    relation_parser.add_argument("--limit", type=int, default=10)
    relation_parser.add_argument("--json", action="store_true")

    triples_parser = subparsers.add_parser("triples", help="List triples for an entity")
    triples_parser.add_argument("--kg", choices=("1", "2"), required=True)
    triples_parser.add_argument("--entity-id", type=int, required=True)
    triples_parser.add_argument("--direction", choices=("head", "tail", "both"), default="both")
    triples_parser.add_argument("--relation-id", type=int, default=None)
    triples_parser.add_argument("--limit", type=int, default=20)
    triples_parser.add_argument("--json", action="store_true")

    search_rel_parser = subparsers.add_parser("search-relations", help="Search relations by text")
    search_rel_parser.add_argument("--kg", choices=("1", "2"), required=True)
    search_rel_parser.add_argument("--query", required=True)
    search_rel_parser.add_argument("--limit", type=int, default=20)
    search_rel_parser.add_argument("--json", action="store_true")

    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve triples by relation id/name and optional entity id")
    retrieve_parser.add_argument("--kg", choices=("1", "2"), required=True)
    retrieve_parser.add_argument("--relation-id", type=int, default=None)
    retrieve_parser.add_argument("--relation-query", default=None)
    retrieve_parser.add_argument("--entity-id", type=int, default=None)
    retrieve_parser.add_argument("--limit", type=int, default=20)
    retrieve_parser.add_argument("--json", action="store_true")

    align_parser = subparsers.add_parser("alignment", help="Look up the aligned entity for a given entity id")
    align_parser.add_argument("--kg", choices=("1", "2"), required=True)
    align_parser.add_argument("--id", type=int, required=True, dest="entity_id")
    align_parser.add_argument("--split", choices=("test", "valid", "ref_ent_ids", "all"), default="all")
    align_parser.add_argument("--json", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="Evaluate DBP15K alignment")
    eval_parser.add_argument("--mode", choices=("raw", "final_model"), default="raw")
    eval_parser.add_argument("--split", choices=("test", "valid", "ref_ent_ids"), default="test")
    eval_parser.add_argument("--batch-size", type=int, default=128)
    eval_parser.add_argument("--top-k", type=int, nargs="+", default=(1, 5, 10))
    eval_parser.add_argument("--neighbor-size", type=int, default=20)
    eval_parser.add_argument("--embedding-name", default="labse", help="Raw embedding family, e.g. labse or bge_m3")
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    eval_parser.add_argument("--json", action="store_true")

    train_parser = subparsers.add_parser("train", help="Train the DBP15K neighbor-graph alignment model")
    train_parser.add_argument("--output-dir", type=Path, default=None)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--embedding-name", choices=("labse", "bge_m3"), default="labse")
    train_parser.add_argument("--epochs", type=int, default=150)
    train_parser.add_argument("--train-batch-size", type=int, default=64)
    train_parser.add_argument("--eval-batch-size", type=int, default=128)
    train_parser.add_argument("--queue-length", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=1e-6)
    train_parser.add_argument("--weight-decay", type=float, default=0.0)
    train_parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    train_parser.add_argument("--temperature", type=float, default=0.08)
    train_parser.add_argument("--momentum", type=float, default=0.9999)
    train_parser.add_argument("--neighbor-size", type=int, default=20)
    train_parser.add_argument("--dropout", type=float, default=0.3)
    train_parser.add_argument("--gat-num", type=int, default=1)
    train_parser.add_argument("--selection-metric", choices=("valid_hits@1", "valid_hits@10", "valid_mrr"), default="valid_hits@1")
    train_parser.add_argument("--seed", type=int, default=37)
    train_parser.add_argument("--log-every-updates", type=int, default=50)
    train_parser.add_argument("--max-train-updates-per-epoch", type=int, default=None)
    train_parser.add_argument("--embedding-model-name", default=None)
    train_parser.add_argument("--embedding-model-dir", type=Path, default=None)
    train_parser.add_argument("--labse-model-name", default=DEFAULT_LABSE_MODEL_NAME)
    train_parser.add_argument("--labse-model-dir", type=Path, default=DEFAULT_LABSE_MODEL_DIR)
    train_parser.add_argument("--hf-endpoint", default=None)
    train_parser.add_argument("--embedding-build-batch-size", type=int, default=32)
    train_parser.add_argument("--embedding-build-max-length", type=int, default=96)
    train_parser.add_argument("--json", action="store_true")

    return parser


def load_dataset(args) -> DBP15KDataset:
    dataset_dir = Path(args.dbp15k_root) / args.dataset
    return DBP15KDataset(dataset_dir=dataset_dir, dataset_name=args.dataset)


def print_payload(payload: object, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            print(f"{key}: {value}")
    else:
        print(payload)


def handle_entity(args, dataset: DBP15KDataset) -> int:
    payload = dataset.describe_entity(
        kg=args.kg,
        entity_id=args.entity_id,
        relation_limit=args.relation_limit,
        triple_limit=args.triple_limit,
    )
    if args.json:
        print_payload(payload, as_json=True)
        return 0

    print(f"Dataset: {payload['dataset']}")
    print(f"KG{payload['kg']} entity {payload['entity_id']}: {payload['name'] or '<EMPTY>'}")
    alignments = payload["alignments"]
    if alignments:
        print("Alignments:")
        for split_name, record in alignments.items():
            print(f"  {split_name}: KG{record['kg']} {record['entity_id']} -> {record['name'] or '<EMPTY>'}")
    else:
        print("Alignments: none")

    relation_summary = payload["relation_summary"]
    if relation_summary:
        print("Top relations:")
        for item in relation_summary:
            print(f"  {item['relation_id']}: {item['relation_name'] or '<EMPTY>'} (count={item['count']})")
    else:
        print("Top relations: none")

    triples = payload["triples"]
    if triples:
        print("Triples:")
        for triple in triples:
            print(
                f"  ({triple['head_id']}:{triple['head_name'] or '<EMPTY>'}) - "
                f"[{triple['relation_id']}:{triple['relation_name'] or '<EMPTY>'}] -> "
                f"({triple['tail_id']}:{triple['tail_name'] or '<EMPTY>'})"
            )
    else:
        print("Triples: none")
    return 0


def handle_relation(args, dataset: DBP15KDataset) -> int:
    relation = dataset.get_relation(args.kg, args.relation_id)
    if relation is None:
        raise KeyError(f"Relation {args.relation_id} not found in KG{args.kg}")
    all_triples = dataset.search_triples_by_relation(args.kg, relation_id=args.relation_id, limit=10**9)
    sample_triples = [dataset.render_triple(triple) for triple in all_triples[: args.limit]]
    payload = {
        "kg": args.kg,
        "relation_id": relation.relation_id,
        "name": relation.name,
        "triple_count": len(all_triples),
        "sample_triples": sample_triples,
    }
    if args.json:
        print_payload(payload, as_json=True)
        return 0

    print(f"KG{args.kg} relation {relation.relation_id}: {relation.name or '<EMPTY>'}")
    print(f"Triple count: {len(all_triples)}")
    for triple in sample_triples:
        print(
            f"  ({triple['head_id']}:{triple['head_name'] or '<EMPTY>'}) - "
            f"[{triple['relation_id']}:{triple['relation_name'] or '<EMPTY>'}] -> "
            f"({triple['tail_id']}:{triple['tail_name'] or '<EMPTY>'})"
        )
    return 0


def handle_triples(args, dataset: DBP15KDataset) -> int:
    triples = dataset.get_triples_for_entity(
        kg=args.kg,
        entity_id=args.entity_id,
        direction=args.direction,
        relation_id=args.relation_id,
    )[: args.limit]
    payload = [dataset.render_triple(triple) for triple in triples]
    if args.json:
        print_payload(payload, as_json=True)
        return 0

    if not payload:
        print("No triples found")
        return 0
    for triple in payload:
        print(
            f"({triple['head_id']}:{triple['head_name'] or '<EMPTY>'}) - "
            f"[{triple['relation_id']}:{triple['relation_name'] or '<EMPTY>'}] -> "
            f"({triple['tail_id']}:{triple['tail_name'] or '<EMPTY>'})"
        )
    return 0


def handle_search_relations(args, dataset: DBP15KDataset) -> int:
    relations = dataset.search_relations(args.kg, args.query, limit=args.limit)
    payload = [{"relation_id": relation.relation_id, "name": relation.name, "kg": relation.kg} for relation in relations]
    print_payload(payload, as_json=args.json)
    return 0


def handle_retrieve(args, dataset: DBP15KDataset) -> int:
    triples = dataset.search_triples_by_relation(
        kg=args.kg,
        relation_id=args.relation_id,
        relation_query=args.relation_query,
        entity_id=args.entity_id,
        limit=args.limit,
    )
    payload = [dataset.render_triple(triple) for triple in triples]
    print_payload(payload, as_json=args.json)
    return 0


def handle_alignment(args, dataset: DBP15KDataset) -> int:
    alignments = dataset.find_alignment(args.kg, args.entity_id, split=args.split)
    payload = {
        "kg": args.kg,
        "entity_id": args.entity_id,
        "alignments": {
            split_name: {
                "kg": record.kg,
                "entity_id": record.entity_id,
                "name": record.name,
            }
            for split_name, record in alignments.items()
        },
    }
    print_payload(payload, as_json=args.json)
    return 0


def handle_eval(args, dataset: DBP15KDataset) -> int:
    if args.mode == "raw":
        result = evaluate_raw_alignment(
            dataset=dataset,
            split=args.split,
            top_k=args.top_k,
            batch_size=args.batch_size,
            embedding_name=args.embedding_name,
        )
    else:
        if args.model_path is None:
            raise RuntimeError("--model-path is required for --mode final_model")
        result = evaluate_final_model_alignment(
            dataset=dataset,
            model_path=args.model_path,
            split=args.split,
            top_k=args.top_k,
            batch_size=args.batch_size,
            device=args.device,
            neighbor_size=args.neighbor_size,
            embedding_name=args.embedding_name,
        )
    print_payload(result.to_dict(), as_json=args.json)
    return 0


def handle_train(args) -> int:
    prepared = prepare_alignment_dataset(dataset=args.dataset)
    dataset_dir = Path(args.dbp15k_root) / args.dataset
    if not dataset_dir.exists() or not (dataset_dir / "triples_1").exists():
        dataset_dir = Path(prepared.target_dataset_dir)
    output_dir = args.output_dir or (DEFAULT_ALIGNMENT_TRAINING_OUTPUT_DIR / f"{args.embedding_name}_neighbor_retrained_manual")

    config = AlignmentTrainingConfig(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
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
    summary = train_alignment_model(config)
    payload = {
        "prepared_dataset": prepared.to_dict(),
        "training_summary": summary.to_dict(),
    }
    print_payload(payload, as_json=args.json)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        return handle_train(args)
    dataset = load_dataset(args)

    if args.command == "entity":
        return handle_entity(args, dataset)
    if args.command == "relation":
        return handle_relation(args, dataset)
    if args.command == "triples":
        return handle_triples(args, dataset)
    if args.command == "search-relations":
        return handle_search_relations(args, dataset)
    if args.command == "retrieve":
        return handle_retrieve(args, dataset)
    if args.command == "alignment":
        return handle_alignment(args, dataset)
    if args.command == "eval":
        return handle_eval(args, dataset)
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
