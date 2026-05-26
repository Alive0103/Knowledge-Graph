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

from alignment.config import DEFAULT_DATASET, DEFAULT_DBP15K_ROOT, DEFAULT_MODEL_PATH, REPO_ROOT
from alignment.dbp15k import DBP15KDataset
from alignment.evaluation import evaluate_final_model_alignment, evaluate_raw_alignment


DEFAULT_OUTPUT_DIR = RETRY_DIR / "output" / "experiment_comparison"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a teacher-facing comparison report for DBP15K alignment methods"
    )
    parser.add_argument("--dbp15k-root", type=Path, default=DEFAULT_DBP15K_ROOT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("test", "valid", "ref_ent_ids"), default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--neighbor-size", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-embedding-name", choices=("labse", "bge_m3"), default="labse")
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--include-bge-m3", action="store_true", help="Evaluate raw_BGE_M3_emb_*.pkl if present")
    return parser


def _round_delta(value: float) -> float:
    return round(float(value), 3)


def _path_status(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.exists(),
    }


def _all_exist(*paths: Path) -> bool:
    return all(path.exists() for path in paths)


def _any_exist(paths: list[Path]) -> bool:
    return any(path.exists() for path in paths)


def _bge_embedding_paths(dataset: DBP15KDataset) -> tuple[Path, Path]:
    return (
        dataset.dataset_dir / "raw_BGE_M3_emb_1.pkl",
        dataset.dataset_dir / "raw_BGE_M3_emb_2.pkl",
    )


def discover_method_inventory(dataset: DBP15KDataset, repo_root: Path) -> list[dict[str, object]]:
    raw_left = dataset.dataset_dir / "raw_LaBSE_emb_1.pkl"
    raw_right = dataset.dataset_dir / "raw_LaBSE_emb_2.pkl"
    bge_left, bge_right = _bge_embedding_paths(dataset)
    final_model_candidates = [
        repo_root / "data" / "models" / "final_model.pth",
        repo_root / "跨语言实体对齐" / "final_model.pth",
        repo_root / "archived" / "SelfKG-original" / "dq" / "log2" / "layers_LaBSE_neighbor" / "final_model.pth",
    ]
    ssl_source_candidates = [
        repo_root / "archived" / "SelfKG-original" / "SelfKG-main" / "model" / "layers_LaBSE_SSL.py",
        repo_root / "archived" / "SelfKG-original" / "SelfKG-main" / "model" / "layers_LaBSE_SSL_DWY.py",
    ]
    raw_labse_ready = _all_exist(raw_left, raw_right)
    raw_bge_ready = _all_exist(bge_left, bge_right)
    final_model_ready = _any_exist(final_model_candidates)
    ssl_source_ready = _any_exist(ssl_source_candidates)

    return [
        {
            "method": "raw_labse",
            "label": "Raw LaBSE baseline",
            "type": "verified" if raw_labse_ready else "missing",
            "description": (
                "直接使用 raw_LaBSE_emb_*.pkl 做最近邻检索。"
                if raw_labse_ready
                else "当前工作区缺少 raw_LaBSE_emb_*.pkl。"
            ),
            "artifacts": [_path_status(raw_left), _path_status(raw_right)],
        },
        {
            "method": "raw_bge_m3",
            "label": "Raw BGE-M3 baseline",
            "type": "verified" if raw_bge_ready else "buildable",
            "description": (
                "使用 BGE-M3 实体名称向量做 raw baseline。"
                if raw_bge_ready
                else "代码已支持生成 raw_BGE_M3_emb_*.pkl，但当前文件未就绪。"
            ),
            "artifacts": [_path_status(bge_left), _path_status(bge_right)],
        },
        {
            "method": "labse_neighbor_final_model",
            "label": "LaBSE + neighbor graph model",
            "type": "verified" if final_model_ready else "missing",
            "description": (
                "仓库内已有基于 LaBSE 训练的 graph 权重 final_model.pth。"
                if final_model_ready
                else "仓库内没有可直接复用的 LaBSE graph 权重。"
            ),
            "artifacts": [_path_status(path) for path in final_model_candidates],
        },
        {
            "method": "bge_m3_neighbor_graph",
            "label": "BGE-M3 + neighbor graph model",
            "type": "buildable",
            "description": "现在已支持通过 retry/run_alignment_training.py --embedding-name bge_m3 从零重训。",
            "artifacts": [],
        },
        {
            "method": "labse_ssl",
            "label": "LaBSE SSL",
            "type": "source_only" if ssl_source_ready else "missing",
            "description": (
                "仓库内可见 SSL 相关源码，但没有现成可评测权重。"
                if ssl_source_ready
                else "当前工作区未找到可直接使用的 LaBSE SSL 源码或权重。"
            ),
            "artifacts": [_path_status(path) for path in ssl_source_candidates],
        },
    ]


def _result_block(method: str, label: str, metrics) -> dict[str, object]:
    return {
        "method": method,
        "label": label,
        "metrics": metrics.to_dict(),
    }


def build_report(
    dataset: DBP15KDataset,
    split: str,
    batch_size: int,
    neighbor_size: int,
    device: str,
    model_path: Path | None,
    model_embedding_name: str,
    model_label: str | None,
    include_bge_m3: bool,
) -> dict[str, object]:
    raw_labse = evaluate_raw_alignment(
        dataset=dataset,
        split=split,
        batch_size=batch_size,
        embedding_name="labse",
    )

    bge_result = None
    bge_left, bge_right = _bge_embedding_paths(dataset)
    if (include_bge_m3 or model_embedding_name == "bge_m3") and bge_left.exists() and bge_right.exists():
        bge_result = evaluate_raw_alignment(
            dataset=dataset,
            split=split,
            batch_size=batch_size,
            embedding_name="bge_m3",
        )

    final_result = None
    if model_path is not None and Path(model_path).exists():
        final_result = evaluate_final_model_alignment(
            dataset=dataset,
            model_path=model_path,
            split=split,
            batch_size=batch_size,
            device=device,
            neighbor_size=neighbor_size,
            embedding_name=model_embedding_name,
        )

    before_method = "raw_labse"
    before_label = "Raw LaBSE baseline"
    before_result = raw_labse
    if model_embedding_name == "bge_m3" and bge_result is not None:
        before_method = "raw_bge_m3"
        before_label = "Raw BGE-M3 baseline"
        before_result = bge_result

    delta = None
    if final_result is not None:
        delta = {
            "mrr": _round_delta(final_result.mrr - before_result.mrr),
            "hits@1": _round_delta(final_result.hits_at.get(1, 0.0) - before_result.hits_at.get(1, 0.0)),
            "hits@5": _round_delta(final_result.hits_at.get(5, 0.0) - before_result.hits_at.get(5, 0.0)),
            "hits@10": _round_delta(final_result.hits_at.get(10, 0.0) - before_result.hits_at.get(10, 0.0)),
        }

    additional_baselines: list[dict[str, object]] = []
    if before_method != "raw_labse":
        additional_baselines.append(_result_block("raw_labse", "Raw LaBSE baseline", raw_labse))
    if bge_result is not None and before_method != "raw_bge_m3":
        additional_baselines.append(_result_block("raw_bge_m3", "Raw BGE-M3 baseline", bge_result))

    resolved_model_label = model_label
    if resolved_model_label is None:
        resolved_model_label = (
            "BGE-M3 + neighbor graph model"
            if model_embedding_name == "bge_m3"
            else "LaBSE + neighbor graph model"
        )

    return {
        "dataset": dataset.dataset_name,
        "split": split,
        "comparison": {
            "before_alignment": _result_block(before_method, before_label, before_result),
            "after_alignment": None
            if final_result is None
            else _result_block(f"{model_embedding_name}_neighbor_graph_model", resolved_model_label, final_result),
            "delta_after_minus_before": delta,
            "additional_raw_baselines": additional_baselines,
        },
        "teacher_note": (
            "对齐前结果默认使用与 graph 模型同一 embedding family 的 raw baseline；"
            "对齐后结果为同一 embedding family 下的 neighbor graph model。"
            "如果启用了 BGE-M3，则也会把未作为主线对比的 raw baseline 一并列出。"
        ),
        "method_inventory": discover_method_inventory(dataset, REPO_ROOT),
    }


def render_markdown(report: dict[str, object]) -> str:
    before = report["comparison"]["before_alignment"]
    after = report["comparison"]["after_alignment"]
    delta = report["comparison"]["delta_after_minus_before"]
    extra_raw = report["comparison"]["additional_raw_baselines"]
    inventory = report["method_inventory"]

    lines = [
        "# DBP15K 实验对比报告",
        "",
        f"- 数据集：`{report['dataset']}`",
        f"- 划分：`{report['split']}`",
        f"- 说明：{report['teacher_note']}",
        "",
        "## 指标对比",
        "",
        "| 方法 | MRR | Hits@1 | Hits@5 | Hits@10 |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| {before['label']} | {before['metrics']['mrr']:.3f} | "
            f"{before['metrics']['hits@1']:.3f} | {before['metrics']['hits@5']:.3f} | "
            f"{before['metrics']['hits@10']:.3f} |"
        ),
    ]

    for item in extra_raw:
        lines.append(
            (
                f"| {item['label']} | {item['metrics']['mrr']:.3f} | "
                f"{item['metrics']['hits@1']:.3f} | {item['metrics']['hits@5']:.3f} | "
                f"{item['metrics']['hits@10']:.3f} |"
            )
        )

    if after is not None:
        lines.append(
            (
                f"| {after['label']} | {after['metrics']['mrr']:.3f} | "
                f"{after['metrics']['hits@1']:.3f} | {after['metrics']['hits@5']:.3f} | "
                f"{after['metrics']['hits@10']:.3f} |"
            )
        )

    if delta is not None:
        lines.extend(
            [
                "",
                "## 对齐前后提升",
                "",
                (
                    f"- MRR 提升 `{delta['mrr']:+.3f}`，"
                    f"Hits@1 提升 `{delta['hits@1']:+.3f}`，"
                    f"Hits@5 提升 `{delta['hits@5']:+.3f}`，"
                    f"Hits@10 提升 `{delta['hits@10']:+.3f}`。"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## 仓库内方法盘点",
            "",
            "| 方法 | 状态 | 说明 |",
            "| --- | --- | --- |",
        ]
    )

    for item in inventory:
        status = {
            "verified": "可直接评测",
            "buildable": "可构建/可重训",
            "source_only": "仅有源码",
            "missing": "当前缺失",
        }.get(item["type"], item["type"])
        lines.append(f"| {item['label']} | {status} | {item['description']} |")

    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 当前主线“对齐前”结果使用 `{before['label']}`。",
            (
                f"- 当前主线“对齐后”结果使用 `{after['label']}`。"
                if after is not None
                else "- 当前未提供可用 graph 模型权重，因此只有 raw baseline。"
            ),
            "- 如果需要横向比较文本向量本身的强弱，可以同时参考额外的 raw baseline。",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset = DBP15KDataset(dataset_dir=Path(args.dbp15k_root) / args.dataset, dataset_name=args.dataset)
    report = build_report(
        dataset=dataset,
        split=args.split,
        batch_size=args.batch_size,
        neighbor_size=args.neighbor_size,
        device=args.device,
        model_path=args.model_path,
        model_embedding_name=args.model_embedding_name,
        model_label=args.model_label,
        include_bge_m3=args.include_bge_m3,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.output_suffix.strip()
    json_path = args.output_dir / f"{args.dataset}_{args.split}_comparison{suffix}.json"
    md_path = args.output_dir / f"{args.dataset}_{args.split}_comparison{suffix}.md"

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(report))

    print(
        json.dumps(
            {
                "json_report": str(json_path),
                "markdown_report": str(md_path),
                "comparison": report["comparison"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
