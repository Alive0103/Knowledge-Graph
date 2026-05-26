from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    RETRY_DIR = Path(__file__).resolve().parents[1]
    if str(RETRY_DIR) not in sys.path:
        sys.path.insert(0, str(RETRY_DIR))
    from kg_rag.config import KgRagConfig
    from kg_rag.service import KgRagService
else:
    from .config import KgRagConfig
    from .service import KgRagService


def _config_from_args(args: argparse.Namespace) -> KgRagConfig:
    payload = {
        "kg_repo_root": args.kg_repo_root,
        "dbp15k_root": args.dbp15k_root,
        "dbp15k_dataset": args.dataset,
        "entity_linking_processed_dir": args.entity_linking_processed_dir,
        "alignment_model_path": args.alignment_model_path,
        "bge_alignment_model_path": args.bge_alignment_model_path,
        "embedding_family": args.embedding_family,
        "es_url": args.es_url,
        "es_index_name": args.es_index_name,
        "enable_alignment_expansion": args.enable_alignment_expansion,
        "default_query_intent": args.query_intent,
        "default_retrieval_mode": args.retrieval_mode,
        "default_top_k": args.top_k,
        "default_relation_limit": args.relation_limit,
        "default_triple_limit": args.triple_limit,
        "default_kg_side": args.kg_side,
    }
    return KgRagConfig.from_dict(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="kg-rag CLI")
    parser.add_argument("--kg-repo-root", default=None)
    parser.add_argument("--dbp15k-root", default=None)
    parser.add_argument("--dataset", default="zh_en")
    parser.add_argument("--entity-linking-processed-dir", default=None)
    parser.add_argument("--alignment-model-path", default=None)
    parser.add_argument("--bge-alignment-model-path", default=None)
    parser.add_argument("--embedding-family", default="labse")
    parser.add_argument("--es-url", default="http://localhost:9200")
    parser.add_argument("--es-index-name", default="data2")
    parser.add_argument("--enable-alignment-expansion", action="store_true")
    parser.add_argument("--query-intent", default="auto")
    parser.add_argument("--retrieval-mode", default="hybrid")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--relation-limit", type=int, default=10)
    parser.add_argument("--triple-limit", type=int, default=10)
    parser.add_argument("--kg-side", default="auto")

    subparsers = parser.add_subparsers(dest="command", required=True)

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("query")

    entity_parser = subparsers.add_parser("entity-detail")
    entity_parser.add_argument("entity_key")

    relation_parser = subparsers.add_parser("relation-detail")
    relation_parser.add_argument("kg")
    relation_parser.add_argument("relation_id", type=int)

    subgraph_parser = subparsers.add_parser("subgraph")
    subgraph_parser.add_argument("--node-label", default="*")
    subgraph_parser.add_argument("--max-nodes", type=int, default=50)
    subgraph_parser.add_argument("--max-depth", type=int, default=1)

    bench_generate = subparsers.add_parser("generate-benchmark")
    bench_generate.add_argument("--output", required=True)
    bench_generate.add_argument("--count", type=int, default=40)
    bench_generate.add_argument("--seed", type=int, default=42)

    bench_eval = subparsers.add_parser("benchmark-eval")
    bench_eval.add_argument("--benchmark", required=True)
    bench_eval.add_argument("--output-dir", required=True)

    official = subparsers.add_parser("official-comparison")
    official.add_argument("--output-dir", required=True)
    official.add_argument("--split", default="test")
    official.add_argument("--device", default="cpu")

    subparsers.add_parser("health")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    service = KgRagService(_config_from_args(args))

    if args.command == "query":
        result = service.query(
            query=args.query,
            query_intent=args.query_intent,
            retrieval_mode=args.retrieval_mode,
            top_k=args.top_k,
            relation_limit=args.relation_limit,
            triple_limit=args.triple_limit,
            enable_alignment_expansion=args.enable_alignment_expansion,
            kg_side=args.kg_side,
        )
    elif args.command == "entity-detail":
        result = service.get_entity_detail(
            entity_key=args.entity_key,
            relation_limit=args.relation_limit,
            triple_limit=args.triple_limit,
            enable_alignment_expansion=args.enable_alignment_expansion,
        )
    elif args.command == "relation-detail":
        result = service.get_relation_detail(args.kg, args.relation_id, triple_limit=args.triple_limit)
    elif args.command == "subgraph":
        result = service.build_subgraph(
            node_label=args.node_label,
            max_nodes=args.max_nodes,
            max_depth=args.max_depth,
            enable_alignment_expansion=args.enable_alignment_expansion,
            kg_side=args.kg_side,
        )
    elif args.command == "generate-benchmark":
        result = service.generate_benchmark(Path(args.output), count=args.count, seed=args.seed)
    elif args.command == "benchmark-eval":
        result = service.run_benchmark_eval(args.benchmark, args.output_dir)
    elif args.command == "official-comparison":
        result = service.run_official_comparison(args.output_dir, split=args.split, device=args.device)
    else:
        result = service.healthcheck()

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
