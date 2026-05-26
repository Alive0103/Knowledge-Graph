#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


RETRY_DIR = Path(__file__).resolve().parent
KG_DIR = RETRY_DIR.parent
WORK_WYY_DIR = KG_DIR / "work_wyy"
SEARCH_SCRIPT = WORK_WYY_DIR / "search_vllm.py"
VENDOR_DIR = RETRY_DIR / "_vendor"
OUTPUT_DIR = RETRY_DIR / "output"
STATE_FILE = OUTPUT_DIR / "work_wyy_entity_linking_eval_state.json"
LAUNCH_FILE = OUTPUT_DIR / "work_wyy_entity_linking_eval_launch.json"
STDOUT_LOG = OUTPUT_DIR / "work_wyy_entity_linking_eval_stdout.log"
STDERR_LOG = OUTPUT_DIR / "work_wyy_entity_linking_eval_stderr.log"

MODE_ARGS = {
    "all": [],
    "vector_only": ["--vector-only"],
    "es_text_only": ["--es-text-only"],
    "llm_only": ["--llm-only"],
    "vector_with_llm_always": ["--vector-llm-always"],
    "vector_with_llm": ["--vector-llm"],
}


def default_python():
    candidates = [
        Path(r"D:\software\anaconda\python.exe"),
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def build_env(args):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["KG_EVAL_STATE_FILE"] = str(Path(args.state_file).resolve())
    env["KG_EVAL_MAX_WORKERS"] = str(args.workers)
    env["KG_EVAL_LLM_WORKERS"] = str(args.llm_workers)
    env["KG_EVAL_LLM_RETRIES"] = str(args.llm_retries)
    env["KG_EVAL_LLM_RETRY_DELAY_SECONDS"] = str(args.llm_retry_delay)
    env["KG_EVAL_STATE_INTERVAL_SECONDS"] = str(args.state_interval)

    if args.limit and args.limit > 0:
        env["KG_EVAL_LIMIT"] = str(args.limit)
    else:
        env.pop("KG_EVAL_LIMIT", None)

    existing_pythonpath = env.get("PYTHONPATH", "")
    vendor_path = str(VENDOR_DIR.resolve())
    if existing_pythonpath:
        env["PYTHONPATH"] = vendor_path + os.pathsep + existing_pythonpath
    else:
        env["PYTHONPATH"] = vendor_path

    return env


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def build_command(args):
    command = [args.python, str(SEARCH_SCRIPT.resolve())]
    command.extend(MODE_ARGS[args.mode])
    return command


def ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def parse_args():
    parser = argparse.ArgumentParser(description="Launch work_wyy entity-linking evaluation.")
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_ARGS.keys()),
        default="all",
        help="Evaluation mode to run. Default runs all 5 modes sequentially.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional query limit for smoke tests.")
    parser.add_argument("--workers", type=int, default=6, help="Thread count for non-LLM modes.")
    parser.add_argument("--llm-workers", type=int, default=2, help="Thread count for LLM modes.")
    parser.add_argument("--llm-retries", type=int, default=3, help="Retry count for each LLM call.")
    parser.add_argument("--llm-retry-delay", type=float, default=5.0, help="Base retry delay in seconds.")
    parser.add_argument("--state-interval", type=float, default=5.0, help="State file refresh interval in seconds.")
    parser.add_argument("--state-file", default=str(STATE_FILE), help="Where to write live progress JSON.")
    parser.add_argument("--launch-file", default=str(LAUNCH_FILE), help="Where to write launch metadata JSON.")
    parser.add_argument("--stdout-log", default=str(STDOUT_LOG), help="Launcher-managed stdout log path.")
    parser.add_argument("--stderr-log", default=str(STDERR_LOG), help="Launcher-managed stderr log path.")
    parser.add_argument("--python", default=default_python(), help="Python executable used for the run.")
    parser.add_argument("--foreground", action="store_true", help="Run in foreground instead of detaching.")
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SEARCH_SCRIPT.exists():
        raise FileNotFoundError(f"Missing search script: {SEARCH_SCRIPT}")

    env = build_env(args)
    command = build_command(args)
    launched_at = datetime.now().isoformat()

    manifest = {
        "launched_at": launched_at,
        "mode": args.mode,
        "limit": args.limit or None,
        "python": args.python,
        "workdir": str(WORK_WYY_DIR.resolve()),
        "command": command,
        "state_file": str(Path(args.state_file).resolve()),
        "stdout_log": str(Path(args.stdout_log).resolve()),
        "stderr_log": str(Path(args.stderr_log).resolve()),
        "foreground": args.foreground,
        "env_overrides": {
            "PYTHONIOENCODING": env["PYTHONIOENCODING"],
            "PYTHONUTF8": env["PYTHONUTF8"],
            "PYTHONPATH": env["PYTHONPATH"],
            "KG_EVAL_MAX_WORKERS": env["KG_EVAL_MAX_WORKERS"],
            "KG_EVAL_LLM_WORKERS": env["KG_EVAL_LLM_WORKERS"],
            "KG_EVAL_LLM_RETRIES": env["KG_EVAL_LLM_RETRIES"],
            "KG_EVAL_LLM_RETRY_DELAY_SECONDS": env["KG_EVAL_LLM_RETRY_DELAY_SECONDS"],
            "KG_EVAL_STATE_INTERVAL_SECONDS": env["KG_EVAL_STATE_INTERVAL_SECONDS"],
            "KG_EVAL_LIMIT": env.get("KG_EVAL_LIMIT"),
        },
    }

    if args.foreground:
        write_json(args.launch_file, {**manifest, "status": "running_foreground"})
        result = subprocess.run(command, cwd=str(WORK_WYY_DIR), env=env)
        write_json(
            args.launch_file,
            {
                **manifest,
                "status": "finished_foreground",
                "returncode": result.returncode,
                "finished_at": datetime.now().isoformat(),
            },
        )
        return result.returncode

    stdout_path = Path(args.stdout_log)
    stderr_path = Path(args.stderr_log)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    env_lines = []
    for key in [
        "PYTHONIOENCODING",
        "PYTHONUTF8",
        "PYTHONPATH",
        "KG_EVAL_STATE_FILE",
        "KG_EVAL_MAX_WORKERS",
        "KG_EVAL_LLM_WORKERS",
        "KG_EVAL_LLM_RETRIES",
        "KG_EVAL_LLM_RETRY_DELAY_SECONDS",
        "KG_EVAL_STATE_INTERVAL_SECONDS",
    ]:
        env_lines.append(f"$env:{key} = {ps_quote(env[key])}")

    if env.get("KG_EVAL_LIMIT"):
        env_lines.append(f"$env:KG_EVAL_LIMIT = {ps_quote(env['KG_EVAL_LIMIT'])}")
    else:
        env_lines.append("Remove-Item Env:KG_EVAL_LIMIT -ErrorAction SilentlyContinue")

    arg_list = ", ".join(ps_quote(arg) for arg in command[1:])
    ps_lines = env_lines + [
        "$ErrorActionPreference = 'Stop'",
        (
            f"$p = Start-Process -FilePath {ps_quote(args.python)} "
            f"-ArgumentList @({arg_list}) "
            f"-WorkingDirectory {ps_quote(str(WORK_WYY_DIR.resolve()))} "
            "-WindowStyle Hidden -PassThru"
        ),
        "$p.Id",
    ]
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", "; ".join(ps_lines)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    pid = int(result.stdout.strip().splitlines()[-1])

    manifest.update(
        {
            "status": "detached",
            "pid": pid,
            "launch_file": str(Path(args.launch_file).resolve()),
        }
    )
    write_json(args.launch_file, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
