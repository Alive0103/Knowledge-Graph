#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

RETRY_DIR = Path(__file__).resolve().parent
REPO_ROOT = RETRY_DIR.parent
if str(RETRY_DIR) not in sys.path:
    sys.path.insert(0, str(RETRY_DIR))

from alignment.config import DEFAULT_DATASET
from alignment.embedding_builder import DEFAULT_BGE_M3_MODEL_DIR, DEFAULT_BGE_M3_MODEL_NAME
from entity_linking.download import DEFAULT_BASE_MODEL_DIR
from entity_linking.training_data import DEFAULT_SUPERVISED_TRAINDATA_DIR
from model_hub import looks_like_model_dir


PREFERRED_ENTITY_LINKING_BASE_MODEL_DIR = (
    REPO_ROOT
    / "archived"
    / "SelfKG-original"
    / "wikidata"
    / "model"
    / "chinese-roberta-wwm-ext-large"
)


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def resolve_default_base_model_dir() -> Path:
    if looks_like_model_dir(PREFERRED_ENTITY_LINKING_BASE_MODEL_DIR):
        return PREFERRED_ENTITY_LINKING_BASE_MODEL_DIR
    return DEFAULT_BASE_MODEL_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete supervised retrain suite from entity linking to alignment comparison"
    )
    parser.add_argument("--python-exe", type=Path, default=Path(sys.executable))
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base-model-dir", type=Path, default=resolve_default_base_model_dir())
    parser.add_argument("--supervised-traindata-dir", type=Path, default=DEFAULT_SUPERVISED_TRAINDATA_DIR)
    parser.add_argument("--entity-linking-epochs", type=int, default=5)
    parser.add_argument("--entity-linking-batch-size", type=int, default=4)
    parser.add_argument("--entity-linking-max-length", type=int, default=256)
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument("--alignment-epochs", type=int, default=150)
    parser.add_argument("--alignment-train-batch-size", type=int, default=64)
    parser.add_argument("--alignment-eval-batch-size", type=int, default=128)
    parser.add_argument("--alignment-queue-length", type=int, default=64)
    parser.add_argument("--alignment-learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--alignment-selection-metric",
        choices=("valid_hits@1", "valid_hits@10", "valid_mrr"),
        default="valid_hits@1",
    )
    parser.add_argument("--alignment-max-train-updates-per-epoch", type=int, default=None)
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("--run-tag", default=None)
    return parser


def write_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def mark_step(state: dict[str, object], name: str, status: str, detail: dict[str, object] | None = None) -> None:
    steps = state.setdefault("steps", {})
    state["current_step"] = name
    steps[name] = {
        "status": status,
        "updated_at": now_text(),
    }
    if detail:
        steps[name]["detail"] = detail


def touch_running_step(state: dict[str, object], name: str, detail: dict[str, object] | None = None) -> None:
    steps = state.setdefault("steps", {})
    step = steps.setdefault(name, {})
    state["current_step"] = name
    step["status"] = "running"
    step["updated_at"] = now_text()
    if detail:
        merged = dict(step.get("detail", {}))
        merged.update(detail)
        step["detail"] = merged


def run_command(
    command: list[str],
    cwd: Path,
    label: str,
    capture_path: Path | None = None,
    *,
    state: dict[str, object] | None = None,
    state_path: Path | None = None,
    step_name: str | None = None,
    heartbeat_seconds: int = 60,
) -> None:
    rendered = " ".join(command)
    print(f"[{now_text()}] {label}")
    print("  " + rendered)

    capture_handle = None
    try:
        if capture_path is None:
            process = subprocess.Popen(command, cwd=cwd)
        else:
            capture_path.parent.mkdir(parents=True, exist_ok=True)
            capture_handle = capture_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=capture_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )

        last_heartbeat = time.monotonic()
        sleep_seconds = max(1, min(max(heartbeat_seconds, 1), 5))
        while True:
            returncode = process.poll()
            if returncode is not None:
                break
            if state is not None and state_path is not None and step_name and heartbeat_seconds > 0:
                now_monotonic = time.monotonic()
                if now_monotonic - last_heartbeat >= heartbeat_seconds:
                    detail = {
                        "command": rendered,
                        "pid": process.pid,
                    }
                    if capture_path is not None:
                        detail["output_file"] = str(capture_path)
                    touch_running_step(state, step_name, detail)
                    write_state(state_path, state)
                    last_heartbeat = now_monotonic
            time.sleep(sleep_seconds)
    finally:
        if capture_handle is not None:
            capture_handle.close()

    if returncode != 0:
        raise RuntimeError(f"Command failed with exit code {returncode}: {rendered}")


def load_json(path: Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def render_metrics_row(label: str, metrics: dict[str, object]) -> str:
    return (
        f"| {label} | {float(metrics['mrr']):.3f} | {float(metrics['hits@1']):.3f} | "
        f"{float(metrics['hits@5']):.3f} | {float(metrics['hits@10']):.3f} |"
    )


def build_combined_summary(
    *,
    suite_dir: Path,
    entity_linking_text_metrics_path: Path,
    entity_linking_vector_metrics_path: Path,
    labse_report_path: Path,
    bge_report_path: Path,
) -> tuple[Path, Path]:
    text_metrics = load_json(entity_linking_text_metrics_path)
    vector_metrics = load_json(entity_linking_vector_metrics_path)
    labse_report = load_json(labse_report_path)
    bge_report = load_json(bge_report_path)

    methods: list[dict[str, object]] = []
    for report in (labse_report, bge_report):
        comparison = dict(report["comparison"])
        methods.append(dict(comparison["before_alignment"]))
        after = comparison.get("after_alignment")
        if after is not None:
            methods.append(dict(after))
        for extra in list(comparison.get("additional_raw_baselines") or []):
            methods.append(dict(extra))

    deduped: dict[str, dict[str, object]] = {}
    for item in methods:
        deduped[str(item["method"])] = item

    ordered_methods: list[dict[str, object]] = []
    for method_name in (
        "raw_labse",
        "labse_neighbor_graph_model",
        "raw_bge_m3",
        "bge_m3_neighbor_graph_model",
    ):
        if method_name in deduped:
            ordered_methods.append(deduped[method_name])

    summary = {
        "generated_at": now_text(),
        "entity_linking": {
            "text_only": text_metrics,
            "vector_only": vector_metrics,
        },
        "alignment_methods": ordered_methods,
        "reports": {
            "labse_comparison": str(labse_report_path),
            "bge_graph_comparison": str(bge_report_path),
        },
    }

    json_path = suite_dir / "combined_summary.json"
    md_path = suite_dir / "combined_summary.md"
    suite_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    lines = [
        "# 完整重训结果汇总",
        "",
        "## 实体链接评测",
        "",
        "| 方法 | MRR | Hits@1 | Hits@5 | Hits@10 |",
        "| --- | ---: | ---: | ---: | ---: |",
        render_metrics_row("ES text_only", text_metrics),
        render_metrics_row("ES vector_only", vector_metrics),
        "",
        "## 实体对齐评测",
        "",
        "| 方法 | MRR | Hits@1 | Hits@5 | Hits@10 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in ordered_methods:
        lines.append(render_metrics_row(str(item["label"]), dict(item["metrics"])))
    lines.extend(
        [
            "",
            "## 结果文件",
            "",
            f"- LaBSE 对比报告: `{labse_report_path}`",
            f"- BGE-M3 graph 对比报告: `{bge_report_path}`",
            f"- 汇总 JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    stamp = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_tag = f"complete_supervised_retrain_{stamp}"
    suite_dir = RETRY_DIR / "output" / suite_tag
    state_path = suite_dir / "state.json"
    labse_run_tag = f"{stamp}_labse"
    bge_run_tag = f"{stamp}_bge_graph"
    bge_graph_output_dir = RETRY_DIR / "output" / "alignment_training" / f"bge_m3_neighbor_retrained_{args.dataset}_{bge_run_tag}"
    bge_graph_model_path = bge_graph_output_dir / "best_model.pth"
    bge_graph_eval_path = suite_dir / "bge_m3_graph_test.json"

    state: dict[str, object] = {
        "started_at": now_text(),
        "status": "running",
        "dataset": args.dataset,
        "device": args.device,
        "run_tag": suite_tag,
        "artifacts": {
            "suite_dir": str(suite_dir),
            "base_model_dir": str(args.base_model_dir),
            "supervised_traindata_dir": str(args.supervised_traindata_dir),
            "bge_graph_output_dir": str(bge_graph_output_dir),
        },
    }
    write_state(state_path, state)

    try:
        labse_command = [
            str(args.python_exe),
            "retry/run_rigorous_full_experiment.py",
            "--dataset",
            args.dataset,
            "--device",
            args.device,
            "--epochs",
            str(args.entity_linking_epochs),
            "--batch-size",
            str(args.entity_linking_batch_size),
            "--max-length",
            str(args.entity_linking_max_length),
            "--base-model-dir",
            str(args.base_model_dir),
            "--entity-linking-training-data-source",
            "supervised",
            "--entity-linking-supervised-traindata-dir",
            str(args.supervised_traindata_dir),
            "--alignment-model-strategy",
            "retrain",
            "--alignment-epochs",
            str(args.alignment_epochs),
            "--alignment-train-batch-size",
            str(args.alignment_train_batch_size),
            "--alignment-eval-batch-size",
            str(args.alignment_eval_batch_size),
            "--alignment-queue-length",
            str(args.alignment_queue_length),
            "--alignment-learning-rate",
            str(args.alignment_learning_rate),
            "--alignment-selection-metric",
            args.alignment_selection_metric,
            "--include-bge-m3",
            "--run-tag",
            labse_run_tag,
        ]
        if args.alignment_max_train_updates_per_epoch is not None:
            labse_command.extend(
                [
                    "--alignment-max-train-updates-per-epoch",
                    str(args.alignment_max_train_updates_per_epoch),
                ]
            )
        if args.hf_endpoint:
            labse_command.extend(["--hf-endpoint", args.hf_endpoint])
        if args.heartbeat_seconds > 0:
            labse_command.extend(["--heartbeat-seconds", str(args.heartbeat_seconds)])

        mark_step(state, "labse_full_pipeline", "running")
        write_state(state_path, state)
        run_command(
            labse_command,
            cwd=REPO_ROOT,
            label="Run supervised entity-linking + LaBSE full retrain pipeline",
            capture_path=suite_dir / "labse_full_pipeline.log",
            state=state,
            state_path=state_path,
            step_name="labse_full_pipeline",
            heartbeat_seconds=args.heartbeat_seconds,
        )
        mark_step(
            state,
            "labse_full_pipeline",
            "completed",
            {
                "run_tag": labse_run_tag,
                "log_file": str(suite_dir / "labse_full_pipeline.log"),
            },
        )
        write_state(state_path, state)

        mark_step(state, "bge_graph_training", "running")
        write_state(state_path, state)
        bge_train_command = [
            str(args.python_exe),
            "retry/run_alignment_training.py",
            "--dataset",
            args.dataset,
            "--device",
            args.device,
            "--embedding-name",
            "bge_m3",
            "--output-dir",
            str(bge_graph_output_dir),
            "--run-tag",
            bge_run_tag,
            "--epochs",
            str(args.alignment_epochs),
            "--train-batch-size",
            str(args.alignment_train_batch_size),
            "--eval-batch-size",
            str(args.alignment_eval_batch_size),
            "--queue-length",
            str(args.alignment_queue_length),
            "--learning-rate",
            str(args.alignment_learning_rate),
            "--selection-metric",
            args.alignment_selection_metric,
            "--embedding-model-name",
            DEFAULT_BGE_M3_MODEL_NAME,
            "--embedding-model-dir",
            str(DEFAULT_BGE_M3_MODEL_DIR),
        ]
        if args.alignment_max_train_updates_per_epoch is not None:
            bge_train_command.extend(
                [
                    "--max-train-updates-per-epoch",
                    str(args.alignment_max_train_updates_per_epoch),
                ]
            )
        if args.hf_endpoint:
            bge_train_command.extend(["--hf-endpoint", args.hf_endpoint])
        run_command(
            bge_train_command,
            cwd=REPO_ROOT,
            label="Train BGE-M3 neighbor-graph alignment model from scratch",
            capture_path=suite_dir / "bge_graph_training.log",
            state=state,
            state_path=state_path,
            step_name="bge_graph_training",
            heartbeat_seconds=args.heartbeat_seconds,
        )
        mark_step(
            state,
            "bge_graph_training",
            "completed",
            {
                "output_dir": str(bge_graph_output_dir),
                "best_model_path": str(bge_graph_model_path),
            },
        )
        write_state(state_path, state)

        mark_step(state, "bge_graph_eval", "running")
        write_state(state_path, state)
        run_command(
            [
                str(args.python_exe),
                "retry/run_alignment.py",
                "--dataset",
                args.dataset,
                "eval",
                "--mode",
                "final_model",
                "--split",
                "test",
                "--device",
                args.device,
                "--embedding-name",
                "bge_m3",
                "--model-path",
                str(bge_graph_model_path),
                "--json",
            ],
            cwd=REPO_ROOT,
            label="Evaluate BGE-M3 neighbor-graph alignment model",
            capture_path=bge_graph_eval_path,
            state=state,
            state_path=state_path,
            step_name="bge_graph_eval",
            heartbeat_seconds=args.heartbeat_seconds,
        )
        mark_step(
            state,
            "bge_graph_eval",
            "completed",
            {
                "result_file": str(bge_graph_eval_path),
                "model_path": str(bge_graph_model_path),
            },
        )
        write_state(state_path, state)

        mark_step(state, "bge_graph_comparison", "running")
        write_state(state_path, state)
        run_command(
            [
                str(args.python_exe),
                "retry/run_experiment_comparison.py",
                "--dataset",
                args.dataset,
                "--split",
                "test",
                "--device",
                args.device,
                "--model-path",
                str(bge_graph_model_path),
                "--model-embedding-name",
                "bge_m3",
                "--model-label",
                "BGE-M3 + neighbor graph model",
                "--output-suffix",
                "_bge_m3_graph",
                "--include-bge-m3",
            ],
            cwd=REPO_ROOT,
            label="Generate BGE-M3 graph comparison report",
            capture_path=suite_dir / "bge_graph_comparison.log",
            state=state,
            state_path=state_path,
            step_name="bge_graph_comparison",
            heartbeat_seconds=args.heartbeat_seconds,
        )
        mark_step(
            state,
            "bge_graph_comparison",
            "completed",
            {
                "log_file": str(suite_dir / "bge_graph_comparison.log"),
            },
        )
        write_state(state_path, state)

        mark_step(state, "combined_summary", "running")
        write_state(state_path, state)
        labse_report_path = RETRY_DIR / "output" / "experiment_comparison" / f"{args.dataset}_test_comparison.json"
        bge_report_path = RETRY_DIR / "output" / "experiment_comparison" / f"{args.dataset}_test_comparison_bge_m3_graph.json"
        summary_json, summary_md = build_combined_summary(
            suite_dir=suite_dir,
            entity_linking_text_metrics_path=RETRY_DIR / "output" / "entity_linking_eval" / "text_only_metrics.json",
            entity_linking_vector_metrics_path=RETRY_DIR / "output" / "entity_linking_eval" / "vector_only_metrics.json",
            labse_report_path=labse_report_path,
            bge_report_path=bge_report_path,
        )
        mark_step(
            state,
            "combined_summary",
            "completed",
            {
                "json_report": str(summary_json),
                "markdown_report": str(summary_md),
            },
        )
        state["status"] = "completed"
        state["completed_at"] = now_text()
        write_state(state_path, state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        state["status"] = "failed"
        state["failed_at"] = now_text()
        state["error"] = str(exc)
        write_state(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
