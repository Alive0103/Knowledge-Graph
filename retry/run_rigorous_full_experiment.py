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
from alignment.embedding_builder import DEFAULT_BGE_M3_MODEL_NAME
from entity_linking.download import DEFAULT_BASE_MODEL_DIR
from entity_linking.training_data import DEFAULT_SUPERVISED_TRAINDATA_DIR


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def default_alignment_model_candidates() -> list[Path]:
    return [
        REPO_ROOT / "data" / "models" / "final_model.pth",
        REPO_ROOT / "跨语言实体对齐" / "final_model.pth",
        REPO_ROOT / "archived" / "SelfKG-original" / "dq" / "log2" / "layers_LaBSE_neighbor" / "final_model.pth",
    ]


def find_existing_alignment_model(embedding_name: str = "labse") -> Path | None:
    if embedding_name != "labse":
        return None
    for candidate in default_alignment_model_candidates():
        if candidate.exists():
            return candidate
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full retry experiment sequentially with strict asset preparation")
    parser.add_argument("--python-exe", type=Path, default=Path(sys.executable))
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL_DIR)
    parser.add_argument("--entity-linking-training-data-source", choices=("auto", "supervised", "weak"), default="auto")
    parser.add_argument("--entity-linking-supervised-traindata-dir", type=Path, default=DEFAULT_SUPERVISED_TRAINDATA_DIR)
    parser.add_argument("--entity-linking-max-train-examples", type=int, default=None)
    parser.add_argument("--entity-linking-max-dev-examples", type=int, default=None)
    parser.add_argument("--entity-linking-max-test-examples", type=int, default=None)
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument("--include-bge-m3", action="store_true")
    parser.add_argument("--labse-model-name", default="sentence-transformers/LaBSE")
    parser.add_argument("--labse-model-dir", type=Path, default=RETRY_DIR / "models" / "alignment_baselines" / "labse")
    parser.add_argument("--alignment-embedding-name", choices=("labse", "bge_m3"), default="labse")
    parser.add_argument("--alignment-embedding-model-name", default=None)
    parser.add_argument("--alignment-embedding-model-dir", type=Path, default=None)
    parser.add_argument("--alignment-model-strategy", choices=("reuse", "train_if_missing", "retrain"), default="train_if_missing")
    parser.add_argument("--alignment-epochs", type=int, default=150)
    parser.add_argument("--alignment-train-batch-size", type=int, default=64)
    parser.add_argument("--alignment-eval-batch-size", type=int, default=128)
    parser.add_argument("--alignment-queue-length", type=int, default=64)
    parser.add_argument("--alignment-learning-rate", type=float, default=1e-6)
    parser.add_argument("--alignment-selection-metric", choices=("valid_hits@1", "valid_hits@10", "valid_mrr"), default="valid_hits@1")
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
        merged_detail = dict(step.get("detail", {}))
        merged_detail.update(detail)
        step["detail"] = merged_detail


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
    rendered_command = " ".join(command)
    print(f"[{now_text()}] {label}")
    print("  " + rendered_command)

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
                        "command": rendered_command,
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
        raise RuntimeError(f"Command failed with exit code {returncode}: {rendered_command}")


def build_entity_linking_training_command(args, training_output_dir: Path) -> list[str]:
    command = [
        str(args.python_exe),
        "retry/run_entity_linking_training.py",
        "--skip-download",
        "--base-model-dir",
        str(args.base_model_dir),
        "--training-data-dir",
        str(RETRY_DIR / "output" / "entity_linking_training"),
        "--training-data-source",
        args.entity_linking_training_data_source,
        "--supervised-traindata-dir",
        str(args.entity_linking_supervised_traindata_dir),
        "--output-dir",
        str(training_output_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--max-length",
        str(args.max_length),
        "--device",
        args.device,
        "--log-level",
        "INFO",
    ]
    if args.entity_linking_max_train_examples is not None:
        command.extend(["--max-train-examples", str(args.entity_linking_max_train_examples)])
    if args.entity_linking_max_dev_examples is not None:
        command.extend(["--max-dev-examples", str(args.entity_linking_max_dev_examples)])
    if args.entity_linking_max_test_examples is not None:
        command.extend(["--max-test-examples", str(args.entity_linking_max_test_examples)])
    return command


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    stamp = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"rigorous_full_{stamp}"
    training_output_dir = RETRY_DIR / "output" / "entity_linking_training" / f"ner_finetuned_distilbert_mbert_{suffix}"
    entity_output_dir = RETRY_DIR / "output" / f"entity_linking_transformer_distilbert_mbert_{suffix}"
    alignment_output_dir = RETRY_DIR / "output" / f"alignment_eval_{suffix}"
    alignment_training_output_dir = RETRY_DIR / "output" / "alignment_training" / f"{args.alignment_embedding_name}_neighbor_retrained_{args.dataset}_{suffix}"
    state_path = RETRY_DIR / "output" / f"rigorous_full_run_{suffix}" / "state.json"

    state: dict[str, object] = {
        "started_at": now_text(),
        "status": "running",
        "dataset": args.dataset,
        "device": args.device,
        "run_tag": suffix,
        "artifacts": {
            "training_output_dir": str(training_output_dir),
            "entity_linking_output_dir": str(entity_output_dir),
            "alignment_output_dir": str(alignment_output_dir),
            "alignment_training_output_dir": str(alignment_training_output_dir),
        },
    }
    write_state(state_path, state)

    try:
        mark_step(state, "prepare_assets", "running")
        write_state(state_path, state)
        prepare_command = [
            str(args.python_exe),
            "retry/run_prepare_experiment_assets.py",
            "--dataset",
            args.dataset,
            "--base-model-dir",
            str(args.base_model_dir),
            "--check-es",
            "--json",
        ]
        if args.hf_endpoint:
            prepare_command.extend(["--hf-endpoint", args.hf_endpoint])
        if args.include_bge_m3:
            prepare_command.append("--prepare-bge-model")
        run_command(
            prepare_command,
            cwd=REPO_ROOT,
            label="Prepare local data/model assets",
            capture_path=alignment_output_dir / "prepare_assets.json",
            state=state,
            state_path=state_path,
            step_name="prepare_assets",
            heartbeat_seconds=args.heartbeat_seconds,
        )
        mark_step(
            state,
            "prepare_assets",
            "completed",
            {"result_file": str(alignment_output_dir / "prepare_assets.json")},
        )
        write_state(state_path, state)

        mark_step(state, "entity_linking_training", "running")
        write_state(state_path, state)
        training_command = build_entity_linking_training_command(args, training_output_dir)
        run_command(
            training_command,
            cwd=REPO_ROOT,
            label="Train entity-linking model from scratch",
            capture_path=alignment_output_dir / "entity_linking_training.log",
            state=state,
            state_path=state_path,
            step_name="entity_linking_training",
            heartbeat_seconds=args.heartbeat_seconds,
        )
        mark_step(
            state,
            "entity_linking_training",
            "completed",
            {"model_dir": str(training_output_dir)},
        )
        write_state(state_path, state)

        mark_step(state, "entity_linking_rebuild", "running")
        write_state(state_path, state)
        run_command(
            [
                str(args.python_exe),
                "retry/run_entity_linking.py",
                "--extractor",
                "transformer",
                "--transformer-ner-model",
                str(training_output_dir),
                "--vectorizer",
                "transformer",
                "--transformer-vector-model",
                str(training_output_dir),
                "--output-dir",
                str(entity_output_dir),
                "--overwrite",
                "--log-level",
                "INFO",
            ],
            cwd=REPO_ROOT,
            label="Rebuild entity-linking processed data",
            state=state,
            state_path=state_path,
            step_name="entity_linking_rebuild",
            heartbeat_seconds=args.heartbeat_seconds,
        )
        mark_step(
            state,
            "entity_linking_rebuild",
            "completed",
            {"output_dir": str(entity_output_dir)},
        )
        write_state(state_path, state)

        mark_step(state, "es_index", "running")
        write_state(state_path, state)
        run_command(
            [
                str(args.python_exe),
                "retry/run_entity_linking_es.py",
                "index",
                "--input-dir",
                str(entity_output_dir),
                "--index-name",
                "data2",
                "--json",
            ],
            cwd=REPO_ROOT,
            label="Index rebuilt entity-linking data into Elasticsearch",
            capture_path=alignment_output_dir / "es_index_summary.json",
            state=state,
            state_path=state_path,
            step_name="es_index",
            heartbeat_seconds=args.heartbeat_seconds,
        )
        mark_step(
            state,
            "es_index",
            "completed",
            {"summary_file": str(alignment_output_dir / "es_index_summary.json")},
        )
        write_state(state_path, state)

        mark_step(state, "es_eval", "running")
        write_state(state_path, state)
        run_command(
            [
                str(args.python_exe),
                "retry/run_entity_linking_es.py",
                "eval",
                "--mode",
                "both",
                "--index-name",
                "data2",
                "--vector-model-dir",
                str(training_output_dir),
                "--output-dir",
                str(RETRY_DIR / "output" / "entity_linking_eval"),
                "--json",
            ],
            cwd=REPO_ROOT,
            label="Evaluate text-only and vector-only entity-linking retrieval",
            capture_path=alignment_output_dir / "entity_linking_eval_summary.json",
            state=state,
            state_path=state_path,
            step_name="es_eval",
            heartbeat_seconds=args.heartbeat_seconds,
        )
        mark_step(
            state,
            "es_eval",
            "completed",
            {"summary_file": str(alignment_output_dir / "entity_linking_eval_summary.json")},
        )
        write_state(state_path, state)

        dataset_alignment_dir = REPO_ROOT / "data" / "processed" / "alignment" / "DBP15K" / args.dataset
        recovered_alignment_dir = RETRY_DIR / "recovered" / "alignment" / "DBP15K" / args.dataset
        raw_labse_ready = (
            (dataset_alignment_dir / "raw_LaBSE_emb_1.pkl").exists()
            and (dataset_alignment_dir / "raw_LaBSE_emb_2.pkl").exists()
        ) or (
            (recovered_alignment_dir / "raw_LaBSE_emb_1.pkl").exists()
            and (recovered_alignment_dir / "raw_LaBSE_emb_2.pkl").exists()
        )
        if raw_labse_ready:
            mark_step(
                state,
                "build_raw_labse_baseline",
                "skipped",
                {"reason": "raw_LaBSE_emb files already exist"},
            )
            write_state(state_path, state)
        else:
            mark_step(state, "build_raw_labse_baseline", "running")
            write_state(state_path, state)
            build_raw_labse_command = [
                str(args.python_exe),
                "retry/run_alignment_embedding_baseline.py",
                "--dataset",
                args.dataset,
                "--model-name",
                args.labse_model_name,
                "--model-dir",
                str(args.labse_model_dir),
                "--embedding-name",
                "labse",
                "--output-prefix",
                "raw_LaBSE_emb",
                "--device",
                args.device,
                "--overwrite",
            ]
            if args.hf_endpoint:
                build_raw_labse_command.extend(["--hf-endpoint", args.hf_endpoint])
            run_command(
                build_raw_labse_command,
                cwd=REPO_ROOT,
                label="Build raw LaBSE alignment baseline because raw_LaBSE_emb files are missing",
                capture_path=alignment_output_dir / "build_raw_labse.json",
                state=state,
                state_path=state_path,
                step_name="build_raw_labse_baseline",
                heartbeat_seconds=args.heartbeat_seconds,
            )
            mark_step(
                state,
                "build_raw_labse_baseline",
                "completed",
                {"result_file": str(alignment_output_dir / "build_raw_labse.json")},
            )
            write_state(state_path, state)

        mark_step(state, "alignment_raw_labse", "running")
        write_state(state_path, state)
        run_command(
            [
                str(args.python_exe),
                "retry/run_alignment.py",
                "eval",
                "--mode",
                "raw",
                "--split",
                "test",
                "--json",
            ],
            cwd=REPO_ROOT,
            label="Evaluate raw LaBSE alignment baseline",
            capture_path=alignment_output_dir / "raw_labse_test.json",
            state=state,
            state_path=state_path,
            step_name="alignment_raw_labse",
            heartbeat_seconds=args.heartbeat_seconds,
        )
        mark_step(
            state,
            "alignment_raw_labse",
            "completed",
            {"result_file": str(alignment_output_dir / "raw_labse_test.json")},
        )
        write_state(state_path, state)

        existing_alignment_model_path = find_existing_alignment_model(args.alignment_embedding_name)
        trained_alignment_model_path = alignment_training_output_dir / "best_model.pth"
        should_train_alignment_model = (
            args.alignment_model_strategy == "retrain"
            or (
                args.alignment_model_strategy == "train_if_missing"
                and existing_alignment_model_path is None
            )
        )

        final_model_path: Path | None = None
        if should_train_alignment_model:
            mark_step(
                state,
                "alignment_model_training",
                "running",
                {"output_dir": str(alignment_training_output_dir)},
            )
            write_state(state_path, state)
            alignment_training_command = [
                str(args.python_exe),
                "retry/run_alignment_training.py",
                "--dataset",
                args.dataset,
                "--device",
                args.device,
                "--embedding-name",
                args.alignment_embedding_name,
                "--output-dir",
                str(alignment_training_output_dir),
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
            ]
            if args.alignment_embedding_model_name:
                alignment_training_command.extend(["--embedding-model-name", args.alignment_embedding_model_name])
            if args.alignment_embedding_model_dir is not None:
                alignment_training_command.extend(["--embedding-model-dir", str(args.alignment_embedding_model_dir)])
            if args.alignment_embedding_name == "labse":
                alignment_training_command.extend(
                    [
                        "--labse-model-name",
                        args.labse_model_name,
                        "--labse-model-dir",
                        str(args.labse_model_dir),
                    ]
                )
            if args.hf_endpoint:
                alignment_training_command.extend(["--hf-endpoint", args.hf_endpoint])
            if args.alignment_max_train_updates_per_epoch is not None:
                alignment_training_command.extend(
                    [
                        "--max-train-updates-per-epoch",
                        str(args.alignment_max_train_updates_per_epoch),
                    ]
                )
            run_command(
                alignment_training_command,
                cwd=REPO_ROOT,
                label="Train LaBSE neighbor-graph alignment model from scratch",
                capture_path=alignment_output_dir / "alignment_training.json",
                state=state,
                state_path=state_path,
                step_name="alignment_model_training",
                heartbeat_seconds=args.heartbeat_seconds,
            )
            mark_step(
                state,
                "alignment_model_training",
                "completed",
                {
                    "result_file": str(alignment_output_dir / "alignment_training.json"),
                    "best_model_path": str(trained_alignment_model_path),
                },
            )
            write_state(state_path, state)
            final_model_path = trained_alignment_model_path
        else:
            detail = {"strategy": args.alignment_model_strategy}
            if existing_alignment_model_path is not None:
                detail["model_path"] = str(existing_alignment_model_path)
            mark_step(
                state,
                "alignment_model_training",
                "skipped",
                detail,
            )
            write_state(state_path, state)
            final_model_path = existing_alignment_model_path

        if final_model_path is None:
            mark_step(
                state,
                "alignment_final_model",
                "skipped",
                {"reason": "no reusable or retrained alignment model is available"},
            )
        else:
            mark_step(state, "alignment_final_model", "running", {"model_path": str(final_model_path)})
            write_state(state_path, state)
            run_command(
                [
                    str(args.python_exe),
                    "retry/run_alignment.py",
                    "eval",
                    "--mode",
                    "final_model",
                    "--split",
                    "test",
                    "--device",
                    args.device,
                    "--embedding-name",
                    args.alignment_embedding_name,
                    "--model-path",
                    str(final_model_path),
                    "--json",
                ],
                cwd=REPO_ROOT,
                label="Evaluate final graph-enhanced alignment model",
                capture_path=alignment_output_dir / "final_model_test.json",
                state=state,
                state_path=state_path,
                step_name="alignment_final_model",
                heartbeat_seconds=args.heartbeat_seconds,
            )
            mark_step(
                state,
                "alignment_final_model",
                "completed",
                {
                    "result_file": str(alignment_output_dir / "final_model_test.json"),
                    "model_path": str(final_model_path),
                },
            )
        write_state(state_path, state)

        if args.include_bge_m3 or args.alignment_embedding_name == "bge_m3":
            mark_step(state, "build_bge_m3_baseline", "running")
            write_state(state_path, state)
            run_command(
                [
                    str(args.python_exe),
                    "retry/run_alignment_embedding_baseline.py",
                    "--dataset",
                    args.dataset,
                    "--model-name",
                    DEFAULT_BGE_M3_MODEL_NAME,
                    "--device",
                    args.device,
                    "--overwrite",
                ],
                cwd=REPO_ROOT,
                label="Build BGE-M3 raw alignment baseline",
                capture_path=alignment_output_dir / "build_bge_m3.json",
                state=state,
                state_path=state_path,
                step_name="build_bge_m3_baseline",
                heartbeat_seconds=args.heartbeat_seconds,
            )
            mark_step(
                state,
                "build_bge_m3_baseline",
                "completed",
                {"result_file": str(alignment_output_dir / "build_bge_m3.json")},
            )
            write_state(state_path, state)

            mark_step(state, "alignment_raw_bge_m3", "running")
            write_state(state_path, state)
            run_command(
                [
                    str(args.python_exe),
                    "retry/run_alignment.py",
                    "eval",
                    "--mode",
                    "raw",
                    "--embedding-name",
                    "bge_m3",
                    "--split",
                    "test",
                    "--json",
                ],
                cwd=REPO_ROOT,
                label="Evaluate raw BGE-M3 alignment baseline",
                capture_path=alignment_output_dir / "raw_bge_m3_test.json",
                state=state,
                state_path=state_path,
                step_name="alignment_raw_bge_m3",
                heartbeat_seconds=args.heartbeat_seconds,
            )
            mark_step(
                state,
                "alignment_raw_bge_m3",
                "completed",
                {"result_file": str(alignment_output_dir / "raw_bge_m3_test.json")},
            )
            write_state(state_path, state)

        mark_step(state, "experiment_comparison", "running")
        write_state(state_path, state)
        comparison_command = [
            str(args.python_exe),
            "retry/run_experiment_comparison.py",
            "--dataset",
            args.dataset,
            "--split",
            "test",
            "--device",
            args.device,
            "--model-embedding-name",
            args.alignment_embedding_name,
        ]
        if final_model_path is not None:
            comparison_command.extend(["--model-path", str(final_model_path)])
        if args.alignment_embedding_name == "bge_m3":
            comparison_command.extend(
                [
                    "--model-label",
                    "BGE-M3 + neighbor graph model",
                    "--output-suffix",
                    "_bge_m3_graph",
                ]
            )
        if args.include_bge_m3 or args.alignment_embedding_name == "bge_m3":
            comparison_command.append("--include-bge-m3")
        run_command(
            comparison_command,
            cwd=REPO_ROOT,
            label="Generate teacher-facing comparison report",
            capture_path=alignment_output_dir / "experiment_comparison_stdout.json",
            state=state,
            state_path=state_path,
            step_name="experiment_comparison",
            heartbeat_seconds=args.heartbeat_seconds,
        )
        mark_step(
            state,
            "experiment_comparison",
            "completed",
            {"result_file": str(alignment_output_dir / "experiment_comparison_stdout.json")},
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
