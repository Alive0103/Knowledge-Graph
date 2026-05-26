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

DEFAULT_PYTHON_EXE = Path(sys.executable)
DEFAULT_BASE_MODEL_DIR = RETRY_DIR / "models" / "entity_linking_base" / "distilbert-base-multilingual-cased"
DEFAULT_TRAINING_DATA_DIR = RETRY_DIR / "output" / "entity_linking_training"
DEFAULT_INITIAL_FINETUNED_DIR = DEFAULT_TRAINING_DATA_DIR / "ner_finetuned_distilbert_mbert"
DEFAULT_RESUMED_FINETUNED_DIR = DEFAULT_TRAINING_DATA_DIR / "ner_finetuned_distilbert_mbert_e2"
DEFAULT_ENTITY_LINKING_OUTPUT_DIR = RETRY_DIR / "output" / "entity_linking_transformer_distilbert_mbert"
DEFAULT_OVERNIGHT_OUTPUT_DIR = RETRY_DIR / "output" / "overnight_recovery"
DEFAULT_ALIGNMENT_OUTPUT_DIR = RETRY_DIR / "output" / "alignment_eval"
DEFAULT_SUPERVISED_TRAINDATA_DIR = REPO_ROOT / "work_wyy" / "data" / "traindata"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wait for the current training run and continue the full retry experiment overnight")
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON_EXE)
    parser.add_argument("--watch-pid", type=int, default=1956)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--target-total-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL_DIR)
    parser.add_argument("--training-data-dir", type=Path, default=DEFAULT_TRAINING_DATA_DIR)
    parser.add_argument("--training-data-source", choices=("auto", "supervised", "weak"), default="auto")
    parser.add_argument("--supervised-traindata-dir", type=Path, default=DEFAULT_SUPERVISED_TRAINDATA_DIR)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-dev-examples", type=int, default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--initial-finetuned-dir", type=Path, default=DEFAULT_INITIAL_FINETUNED_DIR)
    parser.add_argument("--resumed-finetuned-dir", type=Path, default=DEFAULT_RESUMED_FINETUNED_DIR)
    parser.add_argument("--entity-linking-output-dir", type=Path, default=DEFAULT_ENTITY_LINKING_OUTPUT_DIR)
    parser.add_argument("--overnight-output-dir", type=Path, default=DEFAULT_OVERNIGHT_OUTPUT_DIR)
    parser.add_argument("--alignment-output-dir", type=Path, default=DEFAULT_ALIGNMENT_OUTPUT_DIR)
    return parser


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def process_exists(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in (result.stdout or "")
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def load_training_summary(model_dir: Path) -> dict[str, object] | None:
    summary_path = model_dir / "training_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_state(state_path: Path, payload: dict[str, object]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def mark_step(state: dict[str, object], name: str, status: str, detail: dict[str, object] | None = None) -> None:
    steps = state.setdefault("steps", {})
    state["current_step"] = name
    steps[name] = {
        "status": status,
        "updated_at": now_text(),
    }
    if detail:
        steps[name]["detail"] = detail


def run_command(
    command: list[str],
    cwd: Path,
    label: str,
    capture_stdout_path: Path | None = None,
) -> None:
    print(f"[{now_text()}] {label}")
    print("  " + " ".join(command))
    if capture_stdout_path is None:
        result = subprocess.run(command, cwd=cwd, check=False)
    else:
        result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
        capture_stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with open(capture_stdout_path, "w", encoding="utf-8") as handle:
            handle.write(result.stdout or "")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def build_entity_linking_training_command(
    args,
    base_model_dir: Path,
    output_dir: Path,
    epochs: int,
) -> list[str]:
    command = [
        str(args.python_exe),
        "retry/run_entity_linking_training.py",
        "--skip-download",
        "--base-model-dir",
        str(base_model_dir),
        "--training-data-dir",
        str(args.training_data_dir),
        "--training-data-source",
        args.training_data_source,
        "--supervised-traindata-dir",
        str(args.supervised_traindata_dir),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(args.batch_size),
        "--max-length",
        str(args.max_length),
        "--device",
        args.device,
        "--log-level",
        "INFO",
    ]
    if args.max_train_examples is not None:
        command.extend(["--max-train-examples", str(args.max_train_examples)])
    if args.max_dev_examples is not None:
        command.extend(["--max-dev-examples", str(args.max_dev_examples)])
    if args.max_test_examples is not None:
        command.extend(["--max-test-examples", str(args.max_test_examples)])
    return command


def wait_for_initial_training(
    watch_pid: int | None,
    model_dir: Path,
    poll_seconds: int,
    state: dict[str, object],
    state_path: Path,
) -> dict[str, object] | None:
    summary = load_training_summary(model_dir)
    if summary is not None:
        return summary

    while True:
        running = process_exists(watch_pid)
        mark_step(
            state,
            "wait_initial_training",
            "running",
            {
                "watch_pid": watch_pid,
                "model_dir": str(model_dir),
                "process_running": running,
            },
        )
        write_state(state_path, state)

        summary = load_training_summary(model_dir)
        if summary is not None:
            return summary
        if not running:
            return None
        time.sleep(max(poll_seconds, 5))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    state_path = args.overnight_output_dir / "state.json"
    state: dict[str, object] = {
        "started_at": now_text(),
        "status": "running",
        "python_exe": str(args.python_exe),
        "watch_pid": args.watch_pid,
        "artifacts": {},
    }
    write_state(state_path, state)

    try:
        mark_step(state, "wait_initial_training", "running")
        write_state(state_path, state)
        initial_summary = wait_for_initial_training(
            watch_pid=args.watch_pid,
            model_dir=args.initial_finetuned_dir,
            poll_seconds=args.poll_seconds,
            state=state,
            state_path=state_path,
        )

        if initial_summary is not None:
            mark_step(
                state,
                "wait_initial_training",
                "completed",
                {
                    "model_dir": str(args.initial_finetuned_dir),
                    "epochs": initial_summary.get("epochs"),
                    "best_dev_positive_f1": initial_summary.get("best_dev_positive_f1"),
                },
            )
            final_model_dir = args.initial_finetuned_dir
            completed_epochs = int(initial_summary.get("epochs") or 0)
        else:
            mark_step(
                state,
                "wait_initial_training",
                "failed",
                {
                    "reason": "watched process exited before training_summary.json appeared",
                    "model_dir": str(args.initial_finetuned_dir),
                },
            )
            write_state(state_path, state)

            mark_step(state, "fallback_full_training", "running")
            write_state(state_path, state)
            run_command(
                command=build_entity_linking_training_command(
                    args,
                    base_model_dir=args.base_model_dir,
                    output_dir=args.resumed_finetuned_dir,
                    epochs=args.target_total_epochs,
                ),
                cwd=REPO_ROOT,
                label="Fallback full entity-linking training",
            )
            fallback_summary = load_training_summary(args.resumed_finetuned_dir)
            if fallback_summary is None:
                raise RuntimeError("Fallback training finished but training_summary.json is missing")
            mark_step(
                state,
                "fallback_full_training",
                "completed",
                {
                    "model_dir": str(args.resumed_finetuned_dir),
                    "epochs": fallback_summary.get("epochs"),
                    "best_dev_positive_f1": fallback_summary.get("best_dev_positive_f1"),
                },
            )
            final_model_dir = args.resumed_finetuned_dir
            completed_epochs = int(fallback_summary.get("epochs") or 0)

        write_state(state_path, state)

        if completed_epochs < args.target_total_epochs:
            remaining_epochs = args.target_total_epochs - completed_epochs
            mark_step(
                state,
                "resume_training",
                "running",
                {
                    "from_model_dir": str(final_model_dir),
                    "to_model_dir": str(args.resumed_finetuned_dir),
                    "remaining_epochs": remaining_epochs,
                },
            )
            write_state(state_path, state)
            run_command(
                command=build_entity_linking_training_command(
                    args,
                    base_model_dir=final_model_dir,
                    output_dir=args.resumed_finetuned_dir,
                    epochs=remaining_epochs,
                ),
                cwd=REPO_ROOT,
                label="Resume entity-linking training to reach the target total epochs",
            )
            resumed_summary = load_training_summary(args.resumed_finetuned_dir)
            if resumed_summary is None:
                raise RuntimeError("Resumed training finished but training_summary.json is missing")
            mark_step(
                state,
                "resume_training",
                "completed",
                {
                    "model_dir": str(args.resumed_finetuned_dir),
                    "completed_total_epochs": args.target_total_epochs,
                    "last_run_epochs": resumed_summary.get("epochs"),
                    "best_dev_positive_f1": resumed_summary.get("best_dev_positive_f1"),
                },
            )
            final_model_dir = args.resumed_finetuned_dir
        else:
            mark_step(
                state,
                "resume_training",
                "skipped",
                {
                    "reason": "initial training already met target total epochs",
                    "completed_total_epochs": completed_epochs,
                },
            )

        state["artifacts"]["entity_linking_model_dir"] = str(final_model_dir)
        write_state(state_path, state)

        mark_step(state, "entity_linking_rebuild", "running")
        write_state(state_path, state)
        run_command(
            command=[
                str(args.python_exe),
                "retry/run_entity_linking.py",
                "--extractor",
                "transformer",
                "--transformer-ner-model",
                str(final_model_dir),
                "--vectorizer",
                "transformer",
                "--transformer-vector-model",
                str(final_model_dir),
                "--output-dir",
                str(args.entity_linking_output_dir),
                "--overwrite",
                "--log-level",
                "INFO",
            ],
            cwd=REPO_ROOT,
            label="Rebuild processed entity-linking data with the fine-tuned transformer model",
        )
        mark_step(
            state,
            "entity_linking_rebuild",
            "completed",
            {
                "output_dir": str(args.entity_linking_output_dir),
            },
        )
        state["artifacts"]["entity_linking_output_dir"] = str(args.entity_linking_output_dir)
        write_state(state_path, state)

        mark_step(state, "es_index", "running")
        write_state(state_path, state)
        run_command(
            command=[
                str(args.python_exe),
                "retry/run_entity_linking_es.py",
                "index",
                "--input-dir",
                str(args.entity_linking_output_dir),
                "--index-name",
                "data2",
                "--json",
            ],
            cwd=REPO_ROOT,
            label="Import recovered entity-linking data into local Elasticsearch",
            capture_stdout_path=args.overnight_output_dir / "es_index_summary.json",
        )
        mark_step(
            state,
            "es_index",
            "completed",
            {
                "summary_file": str(args.overnight_output_dir / "es_index_summary.json"),
                "index_name": "data2",
            },
        )
        write_state(state_path, state)

        mark_step(state, "es_eval", "running")
        write_state(state_path, state)
        run_command(
            command=[
                str(args.python_exe),
                "retry/run_entity_linking_es.py",
                "eval",
                "--mode",
                "both",
                "--index-name",
                "data2",
                "--vector-model-dir",
                str(final_model_dir),
                "--output-dir",
                str(RETRY_DIR / "output" / "entity_linking_eval"),
                "--json",
            ],
            cwd=REPO_ROOT,
            label="Evaluate text-only and vector-only entity-linking retrieval on find.xlsx",
            capture_stdout_path=args.overnight_output_dir / "entity_linking_eval_summary.json",
        )
        mark_step(
            state,
            "es_eval",
            "completed",
            {
                "summary_file": str(args.overnight_output_dir / "entity_linking_eval_summary.json"),
                "output_dir": str(RETRY_DIR / "output" / "entity_linking_eval"),
            },
        )
        write_state(state_path, state)

        args.alignment_output_dir.mkdir(parents=True, exist_ok=True)

        mark_step(state, "alignment_raw_labse", "running")
        write_state(state_path, state)
        run_command(
            command=[
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
            label="Evaluate raw LaBSE DBP15K alignment baseline",
            capture_stdout_path=args.alignment_output_dir / "raw_labse_test.json",
        )
        mark_step(
            state,
            "alignment_raw_labse",
            "completed",
            {
                "result_file": str(args.alignment_output_dir / "raw_labse_test.json"),
            },
        )
        write_state(state_path, state)

        mark_step(state, "alignment_final_model", "running")
        write_state(state_path, state)
        run_command(
            command=[
                str(args.python_exe),
                "retry/run_alignment.py",
                "eval",
                "--mode",
                "final_model",
                "--split",
                "test",
                "--device",
                args.device,
                "--json",
            ],
            cwd=REPO_ROOT,
            label="Evaluate final DBP15K alignment model",
            capture_stdout_path=args.alignment_output_dir / "final_model_test.json",
        )
        mark_step(
            state,
            "alignment_final_model",
            "completed",
            {
                "result_file": str(args.alignment_output_dir / "final_model_test.json"),
            },
        )
        write_state(state_path, state)

        mark_step(state, "build_bge_m3_baseline", "running")
        write_state(state_path, state)
        try:
            run_command(
                command=[
                    str(args.python_exe),
                    "retry/run_alignment_embedding_baseline.py",
                    "--dataset",
                    "zh_en",
                    "--model-name",
                    "BAAI/bge-m3",
                    "--device",
                    args.device,
                    "--overwrite",
                ],
                cwd=REPO_ROOT,
                label="Build raw BGE-M3 DBP15K baseline embeddings",
                capture_stdout_path=args.alignment_output_dir / "build_bge_m3.json",
            )
            mark_step(
                state,
                "build_bge_m3_baseline",
                "completed",
                {
                    "result_file": str(args.alignment_output_dir / "build_bge_m3.json"),
                },
            )
            mark_step(state, "alignment_raw_bge_m3", "running")
            write_state(state_path, state)
            run_command(
                command=[
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
                label="Evaluate raw BGE-M3 DBP15K alignment baseline",
                capture_stdout_path=args.alignment_output_dir / "raw_bge_m3_test.json",
            )
            mark_step(
                state,
                "alignment_raw_bge_m3",
                "completed",
                {
                    "result_file": str(args.alignment_output_dir / "raw_bge_m3_test.json"),
                },
            )
        except Exception as exc:
            mark_step(
                state,
                "build_bge_m3_baseline",
                "failed",
                {
                    "reason": str(exc),
                },
            )
            mark_step(
                state,
                "alignment_raw_bge_m3",
                "skipped",
                {
                    "reason": "BGE-M3 baseline build failed",
                },
            )
        write_state(state_path, state)

        mark_step(state, "experiment_comparison", "running")
        write_state(state_path, state)
        run_command(
            command=[
                str(args.python_exe),
                "retry/run_experiment_comparison.py",
                "--dataset",
                "zh_en",
                "--split",
                "test",
                "--device",
                args.device,
                "--include-bge-m3",
            ],
            cwd=REPO_ROOT,
            label="Generate the teacher-facing before-vs-after alignment comparison report",
            capture_stdout_path=args.alignment_output_dir / "experiment_comparison_stdout.json",
        )
        mark_step(
            state,
            "experiment_comparison",
            "completed",
            {
                "json_report": str(RETRY_DIR / "output" / "experiment_comparison" / "zh_en_test_comparison.json"),
                "markdown_report": str(RETRY_DIR / "output" / "experiment_comparison" / "zh_en_test_comparison.md"),
            },
        )
        write_state(state_path, state)

        state["completed_at"] = now_text()
        state["status"] = "completed"
        write_state(state_path, state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        state["failed_at"] = now_text()
        state["status"] = "failed"
        state["error"] = str(exc)
        write_state(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
