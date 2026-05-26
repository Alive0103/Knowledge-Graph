#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = REPO_ROOT / "work_wyy"
SCRIPT_PATH = WORK_DIR / "search_vllm.py"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_PATH = Path(os.getenv("KG_REMAINING_WRAPPER_STATE_FILE", OUTPUT_DIR / "work_wyy_remaining_recovery_state.json"))
LOG_PATH = Path(os.getenv("KG_REMAINING_WRAPPER_LOG_FILE", OUTPUT_DIR / "work_wyy_remaining_recovery_python.log"))
WORK_WYY_STATE_PATH = Path(os.getenv("KG_EVAL_STATE_FILE", OUTPUT_DIR / "work_wyy_entity_linking_eval_rigorous_full_state.json"))

MODES = [
    {"name": "llm_only", "flag": "--llm-only"},
    {"name": "vector_with_llm_always", "flag": "--vector-llm-always"},
    {"name": "vector_with_llm", "flag": "--vector-llm"},
]


def now():
    return datetime.now().isoformat()


def write_log(message: str) -> None:
    line = f"[{now()}] {message}\n"
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        write_log(f"failed to read json {path}: {exc}")
        return None


def write_state(payload: dict) -> None:
    payload["updated_at"] = now()
    tmp_path = STATE_PATH.with_suffix(STATE_PATH.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(STATE_PATH)


def get_mode_state(mode_name: str):
    payload = read_json(WORK_WYY_STATE_PATH)
    if not payload:
        return None
    modes = payload.get("modes") or {}
    return modes.get(mode_name)


def mode_completed(mode_name: str):
    mode_state = get_mode_state(mode_name)
    if not mode_state:
        return False, None
    report_file = mode_state.get("report_file")
    if mode_state.get("status") == "completed" and report_file and Path(report_file).exists():
        return True, report_file
    return False, report_file


def main() -> int:
    state = {
        "started_at": now(),
        "updated_at": now(),
        "status": "running",
        "current_mode": None,
        "completed_modes": [],
        "skipped_modes": [],
        "failed_mode": None,
        "script_path": str(SCRIPT_PATH),
        "work_dir": str(WORK_DIR),
        "work_wyy_state_file": str(WORK_WYY_STATE_PATH),
        "log_file": str(LOG_PATH),
    }
    write_state(state)
    write_log("python wrapper started")

    env = os.environ.copy()
    python_exe = sys.executable

    for mode in MODES:
        mode_name = mode["name"]

        completed, report_file = mode_completed(mode_name)
        if completed:
            if mode_name not in state["completed_modes"]:
                state["completed_modes"].append(mode_name)
            if mode_name not in state["skipped_modes"]:
                state["skipped_modes"].append(mode_name)
            write_state(state)
            write_log(f"skip completed mode: {mode_name} report={report_file}")
            continue

        state["current_mode"] = mode_name
        state["status"] = "running"
        write_state(state)
        write_log(f"start mode: {mode_name}")

        result = subprocess.run(
            [python_exe, str(SCRIPT_PATH), mode["flag"]],
            cwd=str(WORK_DIR),
            env=env,
            check=False,
        )

        completed, report_file = mode_completed(mode_name)
        write_log(
            f"mode exited: {mode_name}, returncode={result.returncode}, completed={completed}, report={report_file}"
        )

        if completed:
            if mode_name not in state["completed_modes"]:
                state["completed_modes"].append(mode_name)
            write_state(state)
            write_log(f"mode completed: {mode_name}")
            continue

        state["status"] = "failed"
        state["failed_mode"] = mode_name
        write_state(state)
        write_log(f"wrapper failed on mode: {mode_name}")
        return result.returncode if result.returncode is not None else 1

    state["status"] = "completed"
    state["current_mode"] = None
    state["failed_mode"] = None
    state["finished_at"] = now()
    write_state(state)
    write_log("python wrapper completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
