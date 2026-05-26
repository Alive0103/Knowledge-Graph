#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


RETRY_DIR = Path(__file__).resolve().parent
REPO_ROOT = RETRY_DIR.parent
WORK_DIR = REPO_ROOT / "work_wyy"
OUTPUT_DIR = RETRY_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WRAPPER_SCRIPT = RETRY_DIR / "run_remaining_work_wyy_modes.py"
WRAPPER_STATE_PATH = Path(
    os.getenv("KG_REMAINING_WRAPPER_STATE_FILE", OUTPUT_DIR / "work_wyy_remaining_recovery_state.json")
)
WRAPPER_LOG_PATH = Path(
    os.getenv("KG_REMAINING_WRAPPER_LOG_FILE", OUTPUT_DIR / "work_wyy_remaining_recovery_python.log")
)
WORK_STATE_PATH = Path(
    os.getenv("KG_EVAL_STATE_FILE", OUTPUT_DIR / "work_wyy_entity_linking_eval_rigorous_full_state.json")
)
WATCHER_STATE_PATH = Path(
    os.getenv("KG_WATCHER_STATE_FILE", OUTPUT_DIR / "work_wyy_remaining_recovery_watcher_state.json")
)
WATCHER_LOG_PATH = Path(
    os.getenv("KG_WATCHER_LOG_FILE", OUTPUT_DIR / "work_wyy_remaining_recovery_watcher.log")
)
CHECKPOINT_DIR = Path(
    os.getenv("KG_EVAL_CHECKPOINT_DIR", OUTPUT_DIR / "mode_checkpoints")
)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

PYTHON_EXE = Path(os.getenv("KG_WATCHER_PYTHON_EXE") or sys.executable)
POLL_INTERVAL_SECONDS = max(15, int(os.getenv("KG_WATCHER_POLL_INTERVAL_SECONDS", "60")))
STALE_SECONDS = max(300, int(os.getenv("KG_WATCHER_STALE_SECONDS", "1200")))
RESTART_COOLDOWN_SECONDS = max(30, int(os.getenv("KG_WATCHER_RESTART_COOLDOWN_SECONDS", "180")))
MAX_RESTARTS = max(1, int(os.getenv("KG_WATCHER_MAX_RESTARTS", "20")))
TAIL_BYTES = max(4096, int(os.getenv("KG_WATCHER_TAIL_BYTES", str(64 * 1024))))
TARGET_INDEX = os.getenv("KG_ES_INDEX_NAME", "data2_rigorous_full_20260331")

REQUIRED_MODES = [
    "llm_only",
    "vector_with_llm_always",
    "vector_with_llm",
]

BALANCE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"余额不足",
        r"账户余额不足",
        r"账号余额不足",
        r"可用余额不足",
        r"余额已用尽",
        r"账户已欠费",
        r"请充值",
        r"insufficient\s+balance",
        r"insufficient\s+quota",
        r"quota\s+exceeded",
        r"billing\s+hard\s+limit",
        r"credit\s+limit",
    ]
]

runtime_state = {
    "started_at": datetime.now().isoformat(),
    "updated_at": datetime.now().isoformat(),
    "pid": os.getpid(),
    "status": "starting",
    "restart_count": 0,
    "last_restart_at": None,
    "last_restart_reason": None,
    "stop_reason": None,
    "finished_at": None,
    "paths": {
        "wrapper_script": str(WRAPPER_SCRIPT),
        "wrapper_state_file": str(WRAPPER_STATE_PATH),
        "wrapper_log_file": str(WRAPPER_LOG_PATH),
        "work_state_file": str(WORK_STATE_PATH),
        "watcher_log_file": str(WATCHER_LOG_PATH),
        "checkpoint_dir": str(CHECKPOINT_DIR),
    },
}


def now_iso() -> str:
    return datetime.now().isoformat()


def write_log(message: str) -> None:
    line = f"[{now_iso()}] {message}\n"
    with WATCHER_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


def write_state(**updates) -> None:
    runtime_state.update(updates)
    runtime_state["updated_at"] = now_iso()
    tmp_path = WATCHER_STATE_PATH.with_suffix(WATCHER_STATE_PATH.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(runtime_state, f, ensure_ascii=False, indent=2)
    tmp_path.replace(WATCHER_STATE_PATH)


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        write_log(f"failed to read json {path}: {exc}")
        return None


def parse_iso(timestamp: str | None):
    if not timestamp:
        return None
    try:
        value = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return value


def tail_text(path: Path, max_bytes: int = TAIL_BYTES) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            seek_pos = max(0, size - max_bytes)
            f.seek(seek_pos)
            return f.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        write_log(f"failed to tail {path}: {exc}")
        return ""


def path_mtime(path: Path):
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).astimezone()
    except Exception:
        return None


def load_api_key_from_local_config() -> str | None:
    config_path = WORK_DIR / "local" / "config.py"
    if not config_path.exists():
        return None
    try:
        text = config_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        write_log(f"failed to read local config for api key: {exc}")
        return None

    match = re.search(r'^\s*ZHIPUAI_API_KEY\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def build_child_env() -> dict:
    env = os.environ.copy()

    existing_api_key = env.get("KG_ZHIPU_API_KEY") or env.get("ZHIPUAI_API_KEY")
    if not existing_api_key:
        config_key = load_api_key_from_local_config()
        if config_key:
            existing_api_key = config_key
            write_log("loaded Zhipu API key from local config fallback")
    if not existing_api_key:
        raise RuntimeError("missing KG_ZHIPU_API_KEY/ZHIPUAI_API_KEY and no local config fallback found")

    env["KG_ZHIPU_API_KEY"] = existing_api_key
    env.setdefault("ZHIPUAI_API_KEY", existing_api_key)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    vendor_path = str(RETRY_DIR / "_vendor")
    if env.get("PYTHONPATH"):
        parts = env["PYTHONPATH"].split(os.pathsep)
        if vendor_path not in parts:
            env["PYTHONPATH"] = vendor_path + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = vendor_path

    env["KG_ES_INDEX_NAME"] = TARGET_INDEX
    env["KG_EVAL_STATE_FILE"] = str(WORK_STATE_PATH)
    env["KG_EVAL_CHECKPOINT_DIR"] = str(CHECKPOINT_DIR)
    env["KG_REMAINING_WRAPPER_STATE_FILE"] = str(WRAPPER_STATE_PATH)
    env["KG_REMAINING_WRAPPER_LOG_FILE"] = str(WRAPPER_LOG_PATH)
    env["KG_EVAL_GLM_THINKING"] = env.get("KG_EVAL_GLM_THINKING", "0")
    env["KG_EVAL_LLM_RETRIES"] = env.get("KG_EVAL_LLM_RETRIES", "5")
    env["KG_EVAL_LLM_RETRY_DELAY_SECONDS"] = env.get("KG_EVAL_LLM_RETRY_DELAY_SECONDS", "2")
    env["KG_EVAL_LLM_WORKERS"] = env.get("KG_EVAL_LLM_WORKERS", "1")
    env["KG_EVAL_WORKERS_LLM_ONLY"] = env.get("KG_EVAL_WORKERS_LLM_ONLY", "1")
    env["KG_EVAL_WORKERS_VECTOR_WITH_LLM_ALWAYS"] = env.get("KG_EVAL_WORKERS_VECTOR_WITH_LLM_ALWAYS", "1")
    env["KG_EVAL_WORKERS_VECTOR_WITH_LLM"] = env.get("KG_EVAL_WORKERS_VECTOR_WITH_LLM", "1")
    return env


def run_powershell_json(command: str) -> list[dict]:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
    )
    stdout = (result.stdout or "").strip()
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        write_log(f"powershell command failed rc={result.returncode}: {stderr or stdout}")
        return []
    if not stdout:
        return []
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        write_log(f"failed to parse powershell json: {exc}")
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    return []


def list_eval_processes() -> list[dict]:
    escaped_root = str(REPO_ROOT).replace("'", "''")
    command = (
        f"$root = '{escaped_root}'; "
        "$items = Get-CimInstance Win32_Process | Where-Object { "
        "$_.Name -like 'python*' -and $_.CommandLine -and "
        "$_.CommandLine -like ('*' + $root + '*') -and ("
        "$_.CommandLine -match 'run_remaining_work_wyy_modes\\.py' -or "
        "$_.CommandLine -match 'search_vllm\\.py' -or "
        "$_.CommandLine -match 'run_work_wyy_entity_linking_eval\\.py'"
        ") } | Select-Object ProcessId, CommandLine; "
        "if ($items) { $items | ConvertTo-Json -Compress }"
    )
    processes = run_powershell_json(command)
    for item in processes:
        item["ProcessId"] = int(item["ProcessId"])
    return processes


def process_exists(pid: int | None) -> bool:
    if not pid:
        return False
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
    )
    output = (result.stdout or "").strip()
    return output != "" and "No tasks are running" not in output


def kill_pid(pid: int) -> None:
    if not process_exists(pid):
        return
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
    )


def stop_all_eval_processes(reason: str) -> list[int]:
    processes = list_eval_processes()
    pids = sorted({item["ProcessId"] for item in processes})
    if not pids:
        return []
    for pid in pids:
        kill_pid(pid)
    write_log(f"stopped eval processes for reason={reason}; pids={pids}")
    return pids


def launch_wrapper() -> int:
    env = build_child_env()
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(
        [str(PYTHON_EXE), str(WRAPPER_SCRIPT)],
        cwd=str(RETRY_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    write_log(f"launched wrapper pid={proc.pid}")
    return proc.pid


def mode_completed(mode_state: dict | None) -> bool:
    if not mode_state:
        return False
    report_file = mode_state.get("report_file")
    return mode_state.get("status") == "completed" and bool(report_file) and Path(report_file).exists()


def completed_modes(work_state: dict | None, wrapper_state: dict | None = None) -> list[str]:
    done = set()
    if wrapper_state:
        done.update(wrapper_state.get("completed_modes") or [])
    if work_state:
        modes = work_state.get("modes") or {}
        for mode in REQUIRED_MODES:
            if mode_completed(modes.get(mode)):
                done.add(mode)
    return [mode for mode in REQUIRED_MODES if mode in done]


def all_modes_completed(work_state: dict | None, wrapper_state: dict | None = None) -> bool:
    done = set(completed_modes(work_state, wrapper_state))
    return set(REQUIRED_MODES).issubset(done)


def latest_activity_time(work_state: dict | None) -> datetime | None:
    candidates = []
    if work_state:
        for key in ["updated_at", "started_at", "finished_at"]:
            value = parse_iso(work_state.get(key))
            if value:
                candidates.append(value)
        current_mode = work_state.get("current_mode")
        if current_mode:
            mode_state = (work_state.get("modes") or {}).get(current_mode)
            if mode_state:
                for key in ["updated_at", "started_at", "finished_at"]:
                    value = parse_iso(mode_state.get(key))
                    if value:
                        candidates.append(value)
        for key in ["log_file", "console_log_file"]:
            log_path = work_state.get(key)
            if log_path:
                value = path_mtime(Path(log_path))
                if value:
                    candidates.append(value)
    wrapper_log_mtime = path_mtime(WRAPPER_LOG_PATH)
    if wrapper_log_mtime:
        candidates.append(wrapper_log_mtime)
    return max(candidates) if candidates else None


def recent_log_bundle(work_state: dict | None) -> dict:
    log_paths = []
    if work_state:
        for key in ["log_file", "console_log_file"]:
            raw_path = work_state.get(key)
            if raw_path:
                log_paths.append(Path(raw_path))
    if WRAPPER_LOG_PATH not in log_paths:
        log_paths.append(WRAPPER_LOG_PATH)

    snippets = {}
    for path in log_paths:
        snippets[str(path)] = tail_text(path)
    return snippets


def detect_balance_exhausted(log_texts: dict[str, str]) -> tuple[bool, str | None]:
    for path, text in log_texts.items():
        if not text:
            continue
        for pattern in BALANCE_PATTERNS:
            match = pattern.search(text)
            if match:
                excerpt = text[max(0, match.start() - 80): match.end() + 120].replace("\r", " ").replace("\n", " ")
                excerpt = re.sub(r"\s+", " ", excerpt).strip()
                return True, f"{path}: {excerpt[:300]}"
    return False, None


def should_restart(wrapper_state: dict | None, work_state: dict | None, wrapper_running: bool, worker_running: bool) -> tuple[bool, str | None]:
    if all_modes_completed(work_state, wrapper_state):
        return False, None

    if wrapper_state and wrapper_state.get("status") == "failed":
        return True, f"wrapper_state_failed:{wrapper_state.get('failed_mode') or wrapper_state.get('current_mode')}"

    if work_state and work_state.get("status") == "failed":
        return True, f"work_state_failed:{work_state.get('error') or work_state.get('current_mode')}"

    if not wrapper_running and not worker_running:
        if wrapper_state and wrapper_state.get("status") != "completed":
            return True, "wrapper_and_worker_missing"
        if not wrapper_state:
            return True, "wrapper_state_missing"

    last_activity = latest_activity_time(work_state)
    if last_activity:
        inactive_seconds = (datetime.now().astimezone() - last_activity).total_seconds()
        if inactive_seconds >= STALE_SECONDS:
            return True, f"stale_for_{int(inactive_seconds)}s"

    return False, None


def wrapper_running_info(processes: list[dict]) -> tuple[bool, int | None]:
    for item in processes:
        if "run_remaining_work_wyy_modes.py" in item.get("CommandLine", ""):
            return True, item["ProcessId"]
    return False, None


def worker_running_info(processes: list[dict], work_state: dict | None) -> tuple[bool, int | None]:
    state_pid = None
    if work_state and work_state.get("pid"):
        try:
            state_pid = int(work_state["pid"])
        except (TypeError, ValueError):
            state_pid = None
    if state_pid and process_exists(state_pid):
        return True, state_pid
    for item in processes:
        if "search_vllm.py" in item.get("CommandLine", ""):
            return True, item["ProcessId"]
    return False, state_pid


def restart_wrapper(reason: str) -> bool:
    last_restart_at = parse_iso(runtime_state.get("last_restart_at"))
    if last_restart_at:
        seconds_since_last_restart = (datetime.now().astimezone() - last_restart_at).total_seconds()
        if seconds_since_last_restart < RESTART_COOLDOWN_SECONDS:
            write_log(
                f"restart skipped due to cooldown; reason={reason}; remaining={int(RESTART_COOLDOWN_SECONDS - seconds_since_last_restart)}s"
            )
            return False

    if runtime_state["restart_count"] >= MAX_RESTARTS:
        write_log(f"restart limit reached; refusing restart for reason={reason}")
        write_state(
            status="failed",
            stop_reason="restart_limit_reached",
            finished_at=now_iso(),
        )
        return False

    stop_all_eval_processes(reason=f"restart:{reason}")
    pid = launch_wrapper()
    runtime_state["restart_count"] += 1
    runtime_state["last_restart_at"] = now_iso()
    runtime_state["last_restart_reason"] = reason
    write_state(
        status="running",
        child_wrapper_pid=pid,
        stop_reason=None,
    )
    return True


def build_snapshot(wrapper_state: dict | None, work_state: dict | None, wrapper_running: bool, wrapper_pid: int | None, worker_running: bool, worker_pid: int | None, balance_hit: bool, balance_excerpt: str | None) -> dict:
    mode_state = None
    current_mode = None
    if work_state:
        current_mode = work_state.get("current_mode")
        if current_mode:
            mode_state = (work_state.get("modes") or {}).get(current_mode)

    return {
        "wrapper_running": wrapper_running,
        "wrapper_pid": wrapper_pid,
        "worker_running": worker_running,
        "worker_pid": worker_pid,
        "wrapper_status": (wrapper_state or {}).get("status"),
        "work_status": (work_state or {}).get("status"),
        "current_mode": current_mode,
        "processed_queries": (work_state or {}).get("processed_queries"),
        "total_queries": (work_state or {}).get("total_queries"),
        "current_metrics": (work_state or {}).get("current_metrics"),
        "mode_metrics": (mode_state or {}).get("current_metrics"),
        "completed_modes": completed_modes(work_state, wrapper_state),
        "latest_activity_at": latest_activity_time(work_state).isoformat() if latest_activity_time(work_state) else None,
        "balance_exhausted_detected": balance_hit,
        "balance_excerpt": balance_excerpt,
    }


def main() -> int:
    write_log("watcher started")
    write_state(
        status="running",
        poll_interval_seconds=POLL_INTERVAL_SECONDS,
        stale_seconds=STALE_SECONDS,
        restart_cooldown_seconds=RESTART_COOLDOWN_SECONDS,
        max_restarts=MAX_RESTARTS,
    )

    while True:
        wrapper_state = read_json(WRAPPER_STATE_PATH)
        work_state = read_json(WORK_STATE_PATH)
        processes = list_eval_processes()
        wrapper_running, wrapper_pid = wrapper_running_info(processes)
        worker_running, worker_pid = worker_running_info(processes, work_state)
        logs = recent_log_bundle(work_state)
        balance_hit, balance_excerpt = detect_balance_exhausted(logs)

        snapshot = build_snapshot(
            wrapper_state,
            work_state,
            wrapper_running,
            wrapper_pid,
            worker_running,
            worker_pid,
            balance_hit,
            balance_excerpt,
        )
        write_state(status="running", snapshot=snapshot)

        if balance_hit:
            stopped_pids = stop_all_eval_processes(reason="balance_exhausted")
            write_log(f"balance exhaustion detected; stopping watcher. evidence={balance_excerpt}")
            write_state(
                status="stopped_balance_exhausted",
                stop_reason=balance_excerpt or "balance_exhausted",
                stopped_pids=stopped_pids,
                finished_at=now_iso(),
                snapshot=snapshot,
            )
            return 0

        if all_modes_completed(work_state, wrapper_state):
            write_log("all remaining modes completed; watcher exiting")
            write_state(
                status="completed",
                finished_at=now_iso(),
                stop_reason="all_modes_completed",
                snapshot=snapshot,
            )
            return 0

        restart_needed, reason = should_restart(wrapper_state, work_state, wrapper_running, worker_running)
        if restart_needed and reason:
            write_log(f"restart condition met: {reason}")
            restarted = restart_wrapper(reason)
            if not restarted and runtime_state.get("status") == "failed":
                return 1

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        write_log("watcher interrupted by keyboard")
        write_state(status="interrupted", finished_at=now_iso(), stop_reason="keyboard_interrupt")
        raise
    except SystemExit:
        raise
    except Exception as exc:
        write_log(f"watcher crashed: {exc}")
        write_state(status="failed", finished_at=now_iso(), stop_reason=str(exc))
        raise
