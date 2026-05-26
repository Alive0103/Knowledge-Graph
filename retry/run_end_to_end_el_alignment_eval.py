#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, unquote, urlsplit, urlunsplit

import numpy as np


if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

RETRY_DIR = Path(__file__).resolve().parent
REPO_ROOT = RETRY_DIR.parent
if str(RETRY_DIR) not in sys.path:
    sys.path.insert(0, str(RETRY_DIR))

from alignment.config import DEFAULT_DBP15K_ROOT
from alignment.dbp15k import DBP15KDataset
from alignment.evaluation import encode_alignment_model
from alignment.model import ModelArgs, create_alignment_model, load_checkpoint
from entity_linking.backends import TransformerVectorizer
from entity_linking.es_eval import batch_encode_queries, read_find_pairs, text_search, vector_search
from entity_linking.es_index import DEFAULT_ES_URL, create_es_client


DEFAULT_INDEX_NAME = "data2_rigorous_full_20260331"
DEFAULT_FIND_FILE = REPO_ROOT / "work_wyy" / "data" / "find.xlsx"
DEFAULT_VECTOR_MODEL_DIR = (
    RETRY_DIR
    / "output"
    / "entity_linking_training"
    / "ner_finetuned_distilbert_mbert_rigorous_full_overnight_complete_20260331_001_labse"
)
DEFAULT_LABSE_MODEL_PATH = (
    RETRY_DIR
    / "output"
    / "alignment_training"
    / "labse_neighbor_retrained_zh_en_rigorous_full_overnight_complete_20260331_001_labse"
    / "best_model.pth"
)
DEFAULT_BGE_GRAPH_MODEL_PATH = (
    RETRY_DIR
    / "output"
    / "alignment_training"
    / "bge_m3_neighbor_retrained_zh_en_overnight_complete_20260331_001_bge_graph"
    / "best_model.pth"
)
DEFAULT_REPORT_GLOBS = {
    "llm_only": "evaluation_report_llm_only_*.json",
    "vector_with_llm": "evaluation_report_vector_with_llm_*.json",
    "vector_with_llm_always": "evaluation_report_vector_with_llm_always_*.json",
}
DEFAULT_EL_MODES = (
    "text_only",
    "vector_only",
    "llm_only",
    "vector_with_llm",
    "vector_with_llm_always",
)


@dataclass(frozen=True)
class BenchmarkSample:
    query_index: int
    query: str
    correct_link: str
    kg1_entity_id: int
    kg2_entity_id: int
    primary_split: str


@dataclass(frozen=True)
class AlignmentSpace:
    method: str
    left_ids: np.ndarray
    left_vectors: np.ndarray
    right_ids: np.ndarray
    right_vectors: np.ndarray

    def build_ranker(self) -> Callable[[int], list[int]]:
        left_lookup = {int(entity_id): index for index, entity_id in enumerate(self.left_ids.tolist())}
        cache: dict[int, list[int]] = {}

        def rank_targets(source_entity_id: int) -> list[int]:
            source_entity_id = int(source_entity_id)
            if source_entity_id in cache:
                return cache[source_entity_id]

            left_index = left_lookup.get(source_entity_id)
            if left_index is None:
                cache[source_entity_id] = []
                return cache[source_entity_id]

            source_vector = self.left_vectors[left_index]
            scores = self.right_vectors @ source_vector
            ranking = np.argsort(scores)[::-1]
            cache[source_entity_id] = [int(self.right_ids[index]) for index in ranking]
            return cache[source_entity_id]

        return rank_targets


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Formal end-to-end evaluation for entity linking -> entity alignment on the find.xlsx overlap subset"
    )
    parser.add_argument("--dataset", default="zh_en")
    parser.add_argument("--dbp15k-root", type=Path, default=DEFAULT_DBP15K_ROOT)
    parser.add_argument("--find-file", type=Path, default=DEFAULT_FIND_FILE)
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    parser.add_argument("--es-url", default=DEFAULT_ES_URL)
    parser.add_argument("--vector-model-dir", type=Path, default=DEFAULT_VECTOR_MODEL_DIR)
    parser.add_argument("--labse-model-path", type=Path, default=DEFAULT_LABSE_MODEL_PATH)
    parser.add_argument("--bge-graph-model-path", type=Path, default=DEFAULT_BGE_GRAPH_MODEL_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--neighbor-size", type=int, default=20)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--el-modes", nargs="+", default=list(DEFAULT_EL_MODES))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def canonicalize_wiki_link(link: object) -> str:
    text = str(link or "").strip()
    if not text:
        return ""

    parsed = urlsplit(text)
    if parsed.scheme and parsed.netloc:
        path = unquote(parsed.path or "")
        if path == "/w/index.php":
            title = parse_qs(parsed.query).get("title", [""])[-1]
            if title:
                path = "/wiki/" + unquote(title)
        path = path.rstrip("/")
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, "", ""))
    return text.rstrip("/")


def choose_latest_report(mode: str) -> Path:
    pattern = DEFAULT_REPORT_GLOBS.get(mode)
    if pattern is None:
        raise KeyError(f"No report pattern configured for mode={mode}")
    candidates = sorted((REPO_ROOT / "work_wyy" / "trainlog").glob(pattern), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No evaluation report found for mode={mode} with pattern={pattern}")
    return candidates[-1]


def compute_metrics(ranks: list[int | None]) -> dict[str, object]:
    total = len(ranks) or 1
    reciprocal_rank_sum = 0.0
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0

    for rank in ranks:
        if rank is None:
            continue
        reciprocal_rank_sum += 1.0 / rank
        if rank <= 1:
            hits_at_1 += 1
        if rank <= 5:
            hits_at_5 += 1
        if rank <= 10:
            hits_at_10 += 1

    return {
        "query_count": len(ranks),
        "answered_count": sum(rank is not None for rank in ranks),
        "mrr": round(reciprocal_rank_sum / total, 4),
        "hits@1": round(hits_at_1 / total, 4),
        "hits@5": round(hits_at_5 / total, 4),
        "hits@10": round(hits_at_10 / total, 4),
    }


def rank_for_correct_link(sorted_links: list[str], correct_link: str) -> int | None:
    gold = canonicalize_wiki_link(correct_link)
    for index, candidate in enumerate(sorted_links, start=1):
        if canonicalize_wiki_link(candidate) == gold:
            return index
    return None


def build_benchmark(dataset: DBP15KDataset, find_file: Path) -> tuple[list[BenchmarkSample], dict[str, int]]:
    kg1_link_to_id = {
        canonicalize_wiki_link(entity.name): entity.entity_id
        for entity in dataset.entities["1"].values()
        if canonicalize_wiki_link(entity.name)
    }

    benchmark: list[BenchmarkSample] = []
    counts = {
        "find_rows": 0,
        "kg1_overlap_rows": 0,
        "aligned_overlap_rows": 0,
        "test_rows": 0,
        "valid_rows": 0,
        "ref_ent_ids_rows": 0,
    }

    for query_index, (query, correct_link) in enumerate(read_find_pairs(find_file)):
        counts["find_rows"] += 1
        kg1_entity_id = kg1_link_to_id.get(canonicalize_wiki_link(correct_link))
        if kg1_entity_id is None:
            continue
        counts["kg1_overlap_rows"] += 1

        aligned = dataset.find_alignment("1", kg1_entity_id, split="all")
        if not aligned:
            continue
        counts["aligned_overlap_rows"] += 1

        primary_split = "test" if "test" in aligned else "valid" if "valid" in aligned else "ref_ent_ids"
        counts[f"{primary_split}_rows"] += 1
        benchmark.append(
            BenchmarkSample(
                query_index=query_index,
                query=query,
                correct_link=correct_link,
                kg1_entity_id=kg1_entity_id,
                kg2_entity_id=aligned[primary_split].entity_id,
                primary_split=primary_split,
            )
        )

    return benchmark, counts


def load_report_el_results(mode: str, benchmark: list[BenchmarkSample]) -> tuple[dict[int, dict[str, object]], Path]:
    report_path = choose_latest_report(mode)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    detailed = {int(item["query_index"]): item for item in payload["detailed_results"]}
    filtered = {sample.query_index: detailed[sample.query_index] for sample in benchmark}
    return filtered, report_path


def compute_text_only_results(
    benchmark: list[BenchmarkSample],
    es_url: str,
    index_name: str,
) -> dict[int, dict[str, object]]:
    es = create_es_client(es_url=es_url)
    results: dict[int, dict[str, object]] = {}
    for sample in benchmark:
        hits = text_search(es=es, index_name=index_name, query=sample.query, limit=10)
        results[sample.query_index] = {
            "query_index": sample.query_index,
            "query": sample.query,
            "correct_link": sample.correct_link,
            "sorted_links": [str(hit.get("link") or "") for hit in hits],
        }
    return results


def compute_vector_only_results(
    benchmark: list[BenchmarkSample],
    es_url: str,
    index_name: str,
    vector_model_dir: Path,
) -> dict[int, dict[str, object]]:
    es = create_es_client(es_url=es_url)
    vectorizer = TransformerVectorizer(str(vector_model_dir), dim=1024, batch_size=32)
    query_vectors = batch_encode_queries(vectorizer, [sample.query for sample in benchmark])

    results: dict[int, dict[str, object]] = {}
    for sample, query_vector in zip(benchmark, query_vectors):
        hits = vector_search(
            es=es,
            index_name=index_name,
            query=sample.query,
            vectorizer=vectorizer,
            query_vector=query_vector,
            limit=10,
        )
        results[sample.query_index] = {
            "query_index": sample.query_index,
            "query": sample.query,
            "correct_link": sample.correct_link,
            "sorted_links": [str(hit.get("link") or "") for hit in hits],
        }
    return results


def load_entity_linking_results(
    modes: list[str],
    benchmark: list[BenchmarkSample],
    es_url: str,
    index_name: str,
    vector_model_dir: Path,
) -> tuple[dict[str, dict[int, dict[str, object]]], dict[str, str]]:
    results: dict[str, dict[int, dict[str, object]]] = {}
    sources: dict[str, str] = {}

    for mode in modes:
        if mode == "text_only":
            results[mode] = compute_text_only_results(benchmark=benchmark, es_url=es_url, index_name=index_name)
            sources[mode] = f"es_search:{index_name}"
            continue
        if mode == "vector_only":
            results[mode] = compute_vector_only_results(
                benchmark=benchmark,
                es_url=es_url,
                index_name=index_name,
                vector_model_dir=vector_model_dir,
            )
            sources[mode] = f"vector_search:{vector_model_dir}"
            continue

        report_results, report_path = load_report_el_results(mode=mode, benchmark=benchmark)
        results[mode] = report_results
        sources[mode] = str(report_path)

    return results, sources


def build_alignment_spaces(
    dataset: DBP15KDataset,
    labse_model_path: Path,
    bge_graph_model_path: Path,
    device: str,
    batch_size: int,
    neighbor_size: int,
) -> dict[str, AlignmentSpace]:
    spaces: dict[str, AlignmentSpace] = {}

    left_ids, left_vectors = dataset.get_embedding_matrix("1", embedding_name="labse")
    right_ids, right_vectors = dataset.get_embedding_matrix("2", embedding_name="labse")
    spaces["raw_labse"] = AlignmentSpace(
        method="raw_labse",
        left_ids=left_ids,
        left_vectors=left_vectors,
        right_ids=right_ids,
        right_vectors=right_vectors,
    )

    left_ids, left_vectors = dataset.get_embedding_matrix("1", embedding_name="bge_m3")
    right_ids, right_vectors = dataset.get_embedding_matrix("2", embedding_name="bge_m3")
    spaces["raw_bge_m3"] = AlignmentSpace(
        method="raw_bge_m3",
        left_ids=left_ids,
        left_vectors=left_vectors,
        right_ids=right_ids,
        right_vectors=right_vectors,
    )

    labse_model = create_alignment_model(
        device=device,
        args=ModelArgs(embedding_dim=dataset.get_embedding_dim("1", embedding_name="labse")),
    )
    load_checkpoint(labse_model, model_path=labse_model_path, device=device)
    left_ids, left_vectors = encode_alignment_model(
        dataset=dataset,
        kg="1",
        model=labse_model,
        batch_size=batch_size,
        device=device,
        neighbor_size=neighbor_size,
        embedding_name="labse",
    )
    right_ids, right_vectors = encode_alignment_model(
        dataset=dataset,
        kg="2",
        model=labse_model,
        batch_size=batch_size,
        device=device,
        neighbor_size=neighbor_size,
        embedding_name="labse",
    )
    spaces["labse_neighbor_graph_model"] = AlignmentSpace(
        method="labse_neighbor_graph_model",
        left_ids=left_ids,
        left_vectors=left_vectors,
        right_ids=right_ids,
        right_vectors=right_vectors,
    )

    bge_model = create_alignment_model(
        device=device,
        args=ModelArgs(embedding_dim=dataset.get_embedding_dim("1", embedding_name="bge_m3")),
    )
    load_checkpoint(bge_model, model_path=bge_graph_model_path, device=device)
    left_ids, left_vectors = encode_alignment_model(
        dataset=dataset,
        kg="1",
        model=bge_model,
        batch_size=batch_size,
        device=device,
        neighbor_size=neighbor_size,
        embedding_name="bge_m3",
    )
    right_ids, right_vectors = encode_alignment_model(
        dataset=dataset,
        kg="2",
        model=bge_model,
        batch_size=batch_size,
        device=device,
        neighbor_size=neighbor_size,
        embedding_name="bge_m3",
    )
    spaces["bge_m3_neighbor_graph_model"] = AlignmentSpace(
        method="bge_m3_neighbor_graph_model",
        left_ids=left_ids,
        left_vectors=left_vectors,
        right_ids=right_ids,
        right_vectors=right_vectors,
    )

    return spaces


def mapped_kg1_candidates(sorted_links: list[str], kg1_link_to_id: dict[str, int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for link in sorted_links:
        entity_id = kg1_link_to_id.get(canonicalize_wiki_link(link))
        if entity_id is None or entity_id in seen:
            continue
        seen.add(entity_id)
        ordered.append(entity_id)
    return ordered


def compute_entity_linking_overlap_metrics(
    benchmark: list[BenchmarkSample],
    el_results: dict[int, dict[str, object]],
    kg1_link_to_id: dict[str, int],
) -> dict[str, object]:
    full_ranks: list[int | None] = []
    mapped_ranks: list[int | None] = []
    mapped_coverage = 0

    for sample in benchmark:
        current = el_results[sample.query_index]
        sorted_links = [str(value) for value in current.get("sorted_links", [])]
        full_ranks.append(rank_for_correct_link(sorted_links, sample.correct_link))

        candidate_ids = mapped_kg1_candidates(sorted_links, kg1_link_to_id)
        if candidate_ids:
            mapped_coverage += 1
        try:
            mapped_rank = candidate_ids.index(sample.kg1_entity_id) + 1
        except ValueError:
            mapped_rank = None
        mapped_ranks.append(mapped_rank)

    return {
        "original_link_rank_metrics": compute_metrics(full_ranks),
        "kg1_mapped_rank_metrics": compute_metrics(mapped_ranks),
        "kg1_candidate_coverage": round(mapped_coverage / max(len(benchmark), 1), 4),
        "kg1_candidate_coverage_count": mapped_coverage,
    }


def compose_end_to_end_metrics(
    benchmark: list[BenchmarkSample],
    el_results: dict[int, dict[str, object]],
    kg1_link_to_id: dict[str, int],
    rank_targets: Callable[[int], list[int]],
) -> tuple[dict[str, object], dict[str, object], dict[int, dict[str, object]]]:
    all_ranks: list[int | None] = []
    test_ranks: list[int | None] = []
    details: dict[int, dict[str, object]] = {}

    for sample in benchmark:
        current = el_results[sample.query_index]
        sorted_links = [str(value) for value in current.get("sorted_links", [])]
        source_ids = mapped_kg1_candidates(sorted_links, kg1_link_to_id)

        merged_targets: list[int] = []
        seen_targets: set[int] = set()
        for source_id in source_ids:
            for target_id in rank_targets(source_id):
                if target_id in seen_targets:
                    continue
                seen_targets.add(target_id)
                merged_targets.append(target_id)

        gold_rank = None
        for index, target_id in enumerate(merged_targets, start=1):
            if target_id == sample.kg2_entity_id:
                gold_rank = index
                break

        all_ranks.append(gold_rank)
        if sample.primary_split == "test":
            test_ranks.append(gold_rank)

        details[sample.query_index] = {
            "query_index": sample.query_index,
            "primary_split": sample.primary_split,
            "gold_kg1_entity_id": sample.kg1_entity_id,
            "gold_kg2_entity_id": sample.kg2_entity_id,
            "mapped_kg1_candidates": source_ids,
            "mapped_kg1_candidate_count": len(source_ids),
            "final_gold_rank": gold_rank,
        }

    return compute_metrics(all_ranks), compute_metrics(test_ranks), details


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RETRY_DIR / "output" / f"end_to_end_el_alignment_eval_{stamp}"


def ensure_paths_exist(paths: list[tuple[str, Path]]) -> None:
    missing = [f"{label}: {path}" for label, path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_paths_exist(
        [
            ("DBP15K dataset", args.dbp15k_root / args.dataset),
            ("find file", args.find_file),
            ("vector model dir", args.vector_model_dir),
            ("LaBSE graph model", args.labse_model_path),
            ("BGE-M3 graph model", args.bge_graph_model_path),
        ]
    )

    dataset = DBP15KDataset(dataset_dir=args.dbp15k_root / args.dataset, dataset_name=args.dataset)
    benchmark, benchmark_counts = build_benchmark(dataset=dataset, find_file=args.find_file)
    kg1_link_to_id = {
        canonicalize_wiki_link(entity.name): entity.entity_id
        for entity in dataset.entities["1"].values()
        if canonicalize_wiki_link(entity.name)
    }

    el_results, el_sources = load_entity_linking_results(
        modes=args.el_modes,
        benchmark=benchmark,
        es_url=args.es_url,
        index_name=args.index_name,
        vector_model_dir=args.vector_model_dir,
    )
    alignment_spaces = build_alignment_spaces(
        dataset=dataset,
        labse_model_path=args.labse_model_path,
        bge_graph_model_path=args.bge_graph_model_path,
        device=args.device,
        batch_size=args.eval_batch_size,
        neighbor_size=args.neighbor_size,
    )

    alignment_rankers = {name: space.build_ranker() for name, space in alignment_spaces.items()}

    entity_linking_overlap: dict[str, dict[str, object]] = {}
    end_to_end: dict[str, dict[str, dict[str, object]]] = {}
    per_query_details: dict[str, dict[int, dict[str, object]]] = {}

    for el_mode, mode_results in el_results.items():
        entity_linking_overlap[el_mode] = compute_entity_linking_overlap_metrics(
            benchmark=benchmark,
            el_results=mode_results,
            kg1_link_to_id=kg1_link_to_id,
        )
        end_to_end[el_mode] = {}
        per_query_details[el_mode] = {}

        for ea_method, rank_targets in alignment_rankers.items():
            all_metrics, test_metrics, details = compose_end_to_end_metrics(
                benchmark=benchmark,
                el_results=mode_results,
                kg1_link_to_id=kg1_link_to_id,
                rank_targets=rank_targets,
            )
            end_to_end[el_mode][ea_method] = {
                "all_aligned_overlap": all_metrics,
                "test_only_overlap": test_metrics,
            }
            per_query_details[el_mode][ea_method] = details

    summary = {
        "generated_at": now_text(),
        "dataset": args.dataset,
        "find_file": str(args.find_file),
        "index_name": args.index_name,
        "composition_rule": {
            "entity_linking_to_alignment": "Preserve EL ranking order; keep only EL candidates that map to DBP15K KG1 entities; within each KG1 candidate append the full EA target ranking; deduplicate KG2 targets by first occurrence.",
            "primary_split_selection": "Prefer test, then valid, then ref_ent_ids when the gold Chinese entity appears in multiple alignment files.",
        },
        "benchmark": {
            **benchmark_counts,
            "eligible_query_indices": [sample.query_index for sample in benchmark],
        },
        "entity_linking_sources": el_sources,
        "entity_linking_overlap": entity_linking_overlap,
        "alignment_methods": list(alignment_spaces.keys()),
        "end_to_end": end_to_end,
    }

    summary_path = output_dir / "summary.json"
    details_path = output_dir / "details.json"
    benchmark_path = output_dir / "benchmark.json"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    details_path.write_text(json.dumps(per_query_details, ensure_ascii=False, indent=2), encoding="utf-8")
    benchmark_path.write_text(
        json.dumps([sample.__dict__ for sample in benchmark], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({"summary_file": str(summary_path), "details_file": str(details_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
