#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


RETRY_DIR = Path(__file__).resolve().parent
KG_DIR = RETRY_DIR.parent
WORK_WYY_DIR = KG_DIR / "work_wyy"
LOCAL_DIR = WORK_WYY_DIR / "local"
VENDOR_DIR = RETRY_DIR / "_vendor"
OUTPUT_ROOT = RETRY_DIR / "output" / "work_wyy_dataset_eval_matrix"
ROOT_SEARCH_SCRIPT = WORK_WYY_DIR / "search_vllm.py"
TRAIN_TXT_FALLBACK = KG_DIR / "converted-coreference-linked-with-wiki" / "ner" / "train.txt"
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
SOURCE_GROUPS = {
    "traindata": ("traindata",),
    "ccks": (
        "ccks_json",
        "ccks_validate",
        "ccks_fold0",
        "ccks_fold1",
        "ccks_fold2",
        "ccks_fold3",
        "ccks_fold4",
    ),
    "train_txt": ("train_txt",),
    "msra": ("msra_train", "msra_test"),
}

DEFAULT_CONFIG_ORDER = [
    "config_6_all_data",
    "config_5_all_except_msra",
    "config_4_traindata_ccks",
    "config_1_traindata_only",
    "config_2_ccks_only",
    "config_3_train_txt_only",
]


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def default_python() -> str:
    candidates = [
        Path(r"D:\software\anaconda\python.exe"),
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def prepend_pythonpath(env: dict[str, str], path: Path) -> None:
    existing = env.get("PYTHONPATH", "")
    resolved = str(path.resolve())
    env["PYTHONPATH"] = resolved if not existing else resolved + os.pathsep + existing


def select_eval_state(eval_state: dict[str, Any] | None) -> dict[str, Any] | None:
    if not eval_state:
        return None
    selected = {
        "status": eval_state.get("status"),
        "current_mode": eval_state.get("current_mode"),
        "processed_queries": eval_state.get("processed_queries"),
        "total_queries": eval_state.get("total_queries"),
        "updated_at": eval_state.get("updated_at"),
        "summary_file": eval_state.get("summary_file"),
    }
    if eval_state.get("current_metrics"):
        selected["current_metrics"] = eval_state.get("current_metrics")
    return selected


def render_markdown(state: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# work_wyy 实体链接 6配置 x 5方法矩阵进度")
    lines.append("")
    lines.append(f"- 启动时间: {state.get('started_at')}")
    lines.append(f"- 更新时间: {state.get('updated_at')}")
    lines.append(f"- 当前状态: {state.get('status')}")
    lines.append(f"- 当前配置: {state.get('current_config') or '-'}")
    lines.append(f"- 当前步骤: {state.get('current_step') or '-'}")
    lines.append(f"- ES URL: {state.get('es_url')}")
    lines.append(f"- 评测 query limit: {state.get('limit') or 'FULL'}")
    lines.append("")
    lines.append("| 配置 | 状态 | 文档数 | vector_only | es_text_only | llm_only | vector_llm_always | vector_llm | 备注 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")

    configs = state.get("configs", {})
    for config_id in state.get("config_order", []):
        cfg = configs.get(config_id, {})
        metrics = cfg.get("search_metrics", {}) or {}
        index_status = cfg.get("es_index_status", {}) or {}
        error = cfg.get("error") or ""

        def fmt_metric(mode: str) -> str:
            value = (metrics.get(mode) or {}).get("mrr")
            if value is None:
                return "-"
            return f"{value:.4f}"

        lines.append(
            "| {config} | {status} | {docs} | {m1} | {m2} | {m3} | {m4} | {m5} | {note} |".format(
                config=config_id,
                status=cfg.get("status", "pending"),
                docs=index_status.get("document_count", "-"),
                m1=fmt_metric("vector_only"),
                m2=fmt_metric("es_text_only"),
                m3=fmt_metric("llm_only"),
                m4=fmt_metric("vector_with_llm_always"),
                m5=fmt_metric("vector_with_llm"),
                note=error.replace("\n", " ")[:120] if error else (cfg.get("current_step") or ""),
            )
        )

    return "\n".join(lines) + "\n"


class StateTracker:
    def __init__(self, state_path: Path, report_path: Path, initial_state: dict[str, Any], interval_seconds: int) -> None:
        self.state_path = state_path
        self.report_path = report_path
        self.state = initial_state
        self.interval_seconds = max(5, interval_seconds)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.write()
        self._thread = threading.Thread(target=self._loop, name="matrix-state-heartbeat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.write()

    def _loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self.write()

    def write(self) -> None:
        with self._lock:
            self.state["updated_at"] = now_text()
            self.state["heartbeat_at"] = now_text()
            payload = deepcopy(self.state)
        write_json(self.state_path, payload)
        self.report_path.write_text(render_markdown(payload), encoding="utf-8")

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            self.state.update(kwargs)

    def ensure_config(self, config_id: str, config_info: dict[str, Any]) -> None:
        with self._lock:
            configs = self.state.setdefault("configs", {})
            config_state = configs.setdefault(
                config_id,
                {
                    "config_id": config_id,
                    "name": config_info.get("name"),
                    "description": config_info.get("description"),
                    "status": "pending",
                    "steps": {},
                },
            )
            config_state.setdefault("name", config_info.get("name"))
            config_state.setdefault("description", config_info.get("description"))

    def set_current(self, config_id: str | None, step_name: str | None) -> None:
        with self._lock:
            self.state["current_config"] = config_id
            self.state["current_step"] = step_name

    def update_config(self, config_id: str, **kwargs: Any) -> None:
        with self._lock:
            configs = self.state.setdefault("configs", {})
            config_state = configs.setdefault(config_id, {"config_id": config_id, "steps": {}})
            config_state.update(kwargs)
            config_state["updated_at"] = now_text()

    def update_step(self, config_id: str, step_name: str, status: str, detail: dict[str, Any] | None = None) -> None:
        with self._lock:
            configs = self.state.setdefault("configs", {})
            config_state = configs.setdefault(config_id, {"config_id": config_id, "steps": {}})
            steps = config_state.setdefault("steps", {})
            step_state = steps.setdefault(step_name, {})
            step_state["status"] = status
            step_state["updated_at"] = now_text()
            if detail:
                merged = dict(step_state.get("detail", {}))
                merged.update(detail)
                step_state["detail"] = merged
            config_state["current_step"] = step_name
            config_state["updated_at"] = now_text()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the work_wyy 6-config entity-linking matrix with progress tracking.")
    parser.add_argument("--python", default=default_python(), help="Python executable used for child processes.")
    parser.add_argument("--es-url", default="http://localhost:9200", help="Elasticsearch URL for local execution.")
    parser.add_argument("--config-ids", nargs="*", default=None, help="Optional subset of config ids to run.")
    parser.add_argument("--limit", type=int, default=0, help="Optional query limit for entity-linking evaluation.")
    parser.add_argument("--workers", type=int, default=6, help="Non-LLM worker count for search_vllm.py.")
    parser.add_argument("--llm-workers", type=int, default=1, help="LLM worker count for search_vllm.py.")
    parser.add_argument("--llm-retries", type=int, default=3, help="LLM retry count.")
    parser.add_argument("--llm-retry-delay", type=float, default=5.0, help="LLM retry base delay in seconds.")
    parser.add_argument("--state-interval", type=float, default=5.0, help="Per-query eval state refresh interval in seconds.")
    parser.add_argument("--heartbeat-seconds", type=int, default=30, help="Matrix state heartbeat interval in seconds.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately when one config fails.")
    return parser


def resolve_api_key() -> str:
    for key_name in ("KG_ZHIPU_API_KEY", "ZHIPUAI_API_KEY"):
        value = os.getenv(key_name)
        if value:
            return value

    if str(LOCAL_DIR) not in sys.path:
        sys.path.insert(0, str(LOCAL_DIR))
    local_config = importlib.import_module("config")
    api_key = getattr(local_config, "ZHIPUAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing KG_ZHIPU_API_KEY/ZHIPUAI_API_KEY and no fallback key in work_wyy/local/config.py")
    return api_key


def run_root_eval(
    *,
    python_exe: str,
    config_id: str,
    model_dir: Path,
    run_dir: Path,
    tracker: StateTracker,
    heartbeat_seconds: int,
    limit: int,
    workers: int,
    llm_workers: int,
    llm_retries: int,
    llm_retry_delay: float,
    state_interval: float,
) -> dict[str, Any]:
    config_dir = run_dir / config_id
    config_dir.mkdir(parents=True, exist_ok=True)

    eval_state_path = config_dir / "entity_linking_eval_state.json"
    stdout_log = config_dir / "entity_linking_eval_stdout.log"
    stderr_log = config_dir / "entity_linking_eval_stderr.log"

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["KG_ES_INDEX_NAME"] = f"data_{config_id}"
    env["KG_VECTOR_MODEL_PATH"] = str(model_dir.resolve())
    env["KG_EVAL_STATE_FILE"] = str(eval_state_path.resolve())
    env["KG_EVAL_MAX_WORKERS"] = str(workers)
    env["KG_EVAL_LLM_WORKERS"] = str(llm_workers)
    env["KG_EVAL_LLM_RETRIES"] = str(llm_retries)
    env["KG_EVAL_LLM_RETRY_DELAY_SECONDS"] = str(llm_retry_delay)
    env["KG_EVAL_STATE_INTERVAL_SECONDS"] = str(state_interval)
    if limit > 0:
        env["KG_EVAL_LIMIT"] = str(limit)
    else:
        env.pop("KG_EVAL_LIMIT", None)
    prepend_pythonpath(env, VENDOR_DIR)

    command = [python_exe, str(ROOT_SEARCH_SCRIPT.resolve())]
    tracker.update_step(
        config_id,
        "entity_linking_eval",
        "running",
        {
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "eval_state_file": str(eval_state_path),
            "command": command,
        },
    )

    with stdout_log.open("w", encoding="utf-8") as stdout_handle, stderr_log.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=str(WORK_WYY_DIR),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )

    while True:
        returncode = process.poll()
        eval_state = read_json(eval_state_path)
        tracker.set_current(config_id, "entity_linking_eval")
        tracker.update_step(
            config_id,
            "entity_linking_eval",
            "running" if returncode is None else ("completed" if returncode == 0 else "failed"),
            {
                "pid": process.pid,
                "returncode": returncode,
                "eval_progress": select_eval_state(eval_state),
            },
        )
        tracker.write()
        if returncode is not None:
            break
        time.sleep(max(5, heartbeat_seconds))

    if returncode != 0:
        raise RuntimeError(f"search_vllm.py failed for {config_id} with exit code {returncode}")

    final_eval_state = read_json(eval_state_path) or {}
    return {
        "eval_state_file": str(eval_state_path),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "summary_file": final_eval_state.get("summary_file"),
        "summary": final_eval_state.get("summary", {}),
        "raw_eval_state": final_eval_state,
    }


def has_model_artifacts(model_dir: Path) -> bool:
    required = ("config.json", "tokenizer.json")
    if not model_dir.exists():
        return False
    if not all((model_dir / name).exists() for name in required):
        return False
    return any((model_dir / name).exists() for name in WEIGHT_FILES)


def collect_model_artifacts(model_dir: Path) -> dict[str, Any]:
    return {
        "model_dir": str(model_dir),
        "exists": model_dir.exists(),
        "config_json": (model_dir / "config.json").exists(),
        "tokenizer_json": (model_dir / "tokenizer.json").exists(),
        "label_mapping_json": (model_dir / "label_mapping.json").exists(),
        "model_safetensors": (model_dir / "model.safetensors").exists(),
        "pytorch_model_bin": (model_dir / "pytorch_model.bin").exists(),
        "ready": has_model_artifacts(model_dir),
    }


def ensure_runtime_defaults() -> dict[str, str | None]:
    base_model_candidate = WORK_WYY_DIR / "model" / "chinese-roberta-wwm-ext-large"
    fallback_model_candidate = WORK_WYY_DIR / "model" / "ner_finetuned"

    if has_model_artifacts(base_model_candidate):
        base_model_path = base_model_candidate
    elif has_model_artifacts(fallback_model_candidate):
        base_model_path = fallback_model_candidate
    else:
        base_model_path = base_model_candidate

    if "KG_WORK_WYY_BASE_MODEL_PATH" not in os.environ:
        os.environ["KG_WORK_WYY_BASE_MODEL_PATH"] = str(base_model_path.resolve())

    if "KG_WORK_WYY_TRAIN_TXT_FILE" not in os.environ and TRAIN_TXT_FALLBACK.exists():
        os.environ["KG_WORK_WYY_TRAIN_TXT_FILE"] = str(TRAIN_TXT_FALLBACK.resolve())

    return {
        "base_model_path": os.environ.get("KG_WORK_WYY_BASE_MODEL_PATH"),
        "train_txt_file": os.environ.get("KG_WORK_WYY_TRAIN_TXT_FILE"),
    }


def detect_asset_summary(local_config: Any) -> dict[str, dict[str, Any]]:
    traindata_dir = Path(getattr(local_config, "TRAINDATA_DIR", WORK_WYY_DIR / "data" / "traindata"))
    train_txt_file = Path(getattr(local_config, "TRAIN_TXT_FILE", WORK_WYY_DIR / "data" / "train.txt"))
    ccks_dir = Path(getattr(local_config, "CCKS_NER_DIR", WORK_WYY_DIR / "data" / "ccks_ner" / "militray" / "PreModel_Encoder_CRF"))
    msra_dir = Path(getattr(local_config, "MSRA_DIR", WORK_WYY_DIR / "data" / "nlp_datasets" / "ner" / "msra"))
    base_model_path = Path(getattr(local_config, "BASE_MODEL_PATH", WORK_WYY_DIR / "model" / "chinese-roberta-wwm-ext-large"))

    ccks_train_dir = ccks_dir / "ccks_8_data_v2" / "train"
    ccks_fold_dir = ccks_dir / "data" / "fold0" / "train"
    msra_train = msra_dir / "msra_train_bio.txt"
    msra_test = msra_dir / "msra_test_bio.txt"

    return {
        "base_model": {
            "status": "available" if has_model_artifacts(base_model_path) else "missing",
            "path": str(base_model_path),
        },
        "traindata": {
            "status": "available" if traindata_dir.exists() and any(traindata_dir.glob("*_ner_train.json")) else "missing",
            "path": str(traindata_dir),
        },
        "train_txt": {
            "status": "available" if train_txt_file.exists() else "missing",
            "path": str(train_txt_file),
        },
        "ccks": {
            "status": "available"
            if ccks_train_dir.exists() or ccks_fold_dir.exists() or (ccks_dir / "ccks_8_data_v2" / "validate_data.json").exists()
            else "missing",
            "path": str(ccks_dir),
        },
        "msra": {
            "status": "available" if msra_train.exists() or msra_test.exists() else "missing",
            "path": str(msra_dir),
        },
    }


def effective_switches_for_assets(requested_switches: dict[str, bool], asset_summary: dict[str, dict[str, Any]]) -> tuple[dict[str, bool], list[str]]:
    effective = dict(requested_switches)
    missing_groups: list[str] = []
    for group_name, switch_names in SOURCE_GROUPS.items():
        requested = any(requested_switches.get(name, False) for name in switch_names)
        available = (asset_summary.get(group_name) or {}).get("status") == "available"
        if requested and not available:
            missing_groups.append(group_name)
            for name in switch_names:
                effective[name] = False
    return effective, missing_groups


def enabled_switch_names(switches: dict[str, bool]) -> list[str]:
    return [name for name, enabled in switches.items() if enabled]


def switch_signature(switches: dict[str, bool]) -> tuple[str, ...]:
    return tuple(sorted(enabled_switch_names(switches)))


def apply_model_env(model_dir: Path) -> None:
    resolved = str(model_dir.resolve())
    os.environ["KG_FINETUNED_MODEL_PATH"] = resolved
    os.environ["KG_VECTOR_MODEL_PATH"] = resolved


def copy_config_result(tracker: StateTracker, source_config_id: str, target_config_id: str, note: str) -> None:
    source_state = deepcopy((tracker.state.get("configs", {}) or {}).get(source_config_id, {}))
    copied_steps = source_state.get("steps", {})
    tracker.update_config(
        target_config_id,
        status="aliased",
        note=note,
        aliased_to=source_config_id,
        current_step="aliased",
        model_dir=source_state.get("model_dir"),
        model_artifacts=source_state.get("model_artifacts"),
        ner_metrics=source_state.get("ner_metrics"),
        es_index_status=source_state.get("es_index_status"),
        search_metrics=source_state.get("search_metrics"),
        eval_artifacts=source_state.get("eval_artifacts"),
        steps=deepcopy(copied_steps),
        completed_at=now_text(),
    )


def build_preflight_plan(
    config_order: list[str],
    dataset_configs: dict[str, dict[str, Any]],
    asset_summary: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, list[str]]]:
    plan: dict[str, dict[str, Any]] = {}
    runnable_order: list[str] = []
    alias_targets: dict[str, list[str]] = {}
    canonical_by_signature: dict[tuple[str, ...], str] = {}

    for config_id in config_order:
        config_info = dataset_configs[config_id]
        requested_switches = dict(config_info["switches"])
        effective_switches, missing_groups = effective_switches_for_assets(requested_switches, asset_summary)
        requested_enabled_sources = enabled_switch_names(requested_switches)
        effective_enabled_sources = enabled_switch_names(effective_switches)
        signature = switch_signature(effective_switches)

        note_parts: list[str] = []
        if missing_groups:
            note_parts.append(f"missing source groups: {', '.join(missing_groups)}")
        if effective_enabled_sources != requested_enabled_sources:
            note_parts.append(f"effective sources: {', '.join(effective_enabled_sources) or 'none'}")

        plan_entry = {
            "config_id": config_id,
            "requested_switches": requested_switches,
            "effective_switches": effective_switches,
            "requested_enabled_sources": requested_enabled_sources,
            "effective_enabled_sources": effective_enabled_sources,
            "missing_requested_sources": missing_groups,
            "signature": list(signature),
            "note": "; ".join(note_parts) if note_parts else None,
            "plan_status": "run",
            "alias_of": None,
        }

        if not effective_enabled_sources:
            plan_entry["plan_status"] = "skipped_unavailable"
            plan[config_id] = plan_entry
            continue

        canonical_config_id = canonical_by_signature.get(signature)
        if canonical_config_id is None:
            canonical_by_signature[signature] = config_id
            runnable_order.append(config_id)
            plan_entry["canonical_config_id"] = config_id
        else:
            plan_entry["plan_status"] = "alias"
            plan_entry["alias_of"] = canonical_config_id
            plan_entry["canonical_config_id"] = canonical_config_id
            alias_targets.setdefault(canonical_config_id, []).append(config_id)

        plan[config_id] = plan_entry

    return plan, runnable_order, alias_targets


def main() -> int:
    args = build_parser().parse_args()
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / run_tag
    state_path = OUTPUT_ROOT / "state.json"
    report_path = OUTPUT_ROOT / "state.md"
    run_summary_path = run_dir / "summary.json"

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    os.environ["KG_ES_URL"] = args.es_url
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    runtime_defaults = ensure_runtime_defaults()
    prepend_pythonpath(os.environ, VENDOR_DIR)

    if str(LOCAL_DIR) not in sys.path:
        sys.path.insert(0, str(LOCAL_DIR))
    if str(WORK_WYY_DIR) not in sys.path:
        sys.path.append(str(WORK_WYY_DIR))

    api_key = resolve_api_key()
    os.environ["KG_ZHIPU_API_KEY"] = api_key

    auto_pipeline = importlib.import_module("auto_pipeline")
    local_config = importlib.import_module("config")
    asset_summary = detect_asset_summary(local_config)

    config_order = list(args.config_ids or DEFAULT_CONFIG_ORDER)
    unknown = [config_id for config_id in config_order if config_id not in auto_pipeline.DATASET_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown config ids: {', '.join(unknown)}")
    preflight_plan, runnable_order, alias_targets = build_preflight_plan(
        config_order,
        auto_pipeline.DATASET_CONFIGS,
        asset_summary,
    )

    config_backup_path = run_dir / "local_config.backup.py"
    shutil.copy2(LOCAL_DIR / "config.py", config_backup_path)

    tracker = StateTracker(
        state_path=state_path,
        report_path=report_path,
        initial_state={
            "run_tag": run_tag,
            "started_at": now_text(),
            "updated_at": now_text(),
            "status": "running",
            "es_url": args.es_url,
            "limit": args.limit or None,
            "workers": args.workers,
            "llm_workers": args.llm_workers,
            "llm_retries": args.llm_retries,
            "config_order": config_order,
            "current_config": None,
            "current_step": None,
            "run_dir": str(run_dir),
            "summary_json": str(run_summary_path),
            "runtime_defaults": runtime_defaults,
            "asset_summary": asset_summary,
            "preflight_plan": preflight_plan,
            "runnable_order": runnable_order,
            "configs": {},
        },
        interval_seconds=args.heartbeat_seconds,
    )

    tracker.start()

    try:
        for config_id in config_order:
            config_info = auto_pipeline.DATASET_CONFIGS[config_id]
            plan_entry = preflight_plan[config_id]
            tracker.ensure_config(config_id, config_info)
            tracker.update_config(
                config_id,
                status="pending",
                error=None,
                note=plan_entry.get("note"),
                requested_enabled_sources=plan_entry.get("requested_enabled_sources"),
                effective_enabled_sources=plan_entry.get("effective_enabled_sources"),
                missing_requested_sources=plan_entry.get("missing_requested_sources"),
                canonical_config_id=plan_entry.get("canonical_config_id"),
                alias_of=plan_entry.get("alias_of"),
                plan_status=plan_entry.get("plan_status"),
            )

            if plan_entry["plan_status"] == "skipped_unavailable":
                tracker.set_current(config_id, "skip_unavailable")
                tracker.update_step(
                    config_id,
                    "skip_unavailable",
                    "completed",
                    {"missing_source_groups": plan_entry.get("missing_requested_sources")},
                )
                tracker.update_config(
                    config_id,
                    status="skipped_unavailable",
                    completed_at=now_text(),
                    note="no runnable sources remain after asset check",
                )
            elif plan_entry["plan_status"] == "alias":
                tracker.update_config(
                    config_id,
                    status="pending_alias",
                    note=(plan_entry.get("note") or "") + (
                        f"; waiting for canonical {plan_entry['alias_of']}" if plan_entry.get("alias_of") else ""
                    ),
                )
            else:
                tracker.update_config(config_id, status="queued")

        tracker.write()

        for config_id in runnable_order:
            config_info = auto_pipeline.DATASET_CONFIGS[config_id]
            plan_entry = preflight_plan[config_id]
            effective_switches = dict(plan_entry["effective_switches"])
            effective_enabled_sources = list(plan_entry["effective_enabled_sources"])

            tracker.update_config(config_id, status="running", started_at=now_text(), error=None)

            tracker.set_current(config_id, "update_data_sources")
            tracker.update_step(config_id, "update_data_sources", "running")

            try:
                auto_pipeline.update_config_data_sources(effective_switches)
                tracker.update_step(
                    config_id,
                    "update_data_sources",
                    "completed",
                    {"enabled_sources": effective_enabled_sources},
                )

                tracker.set_current(config_id, "check_prerequisites")
                tracker.update_step(config_id, "check_prerequisites", "running")
                prerequisites_ok = auto_pipeline.check_prerequisites()
                tracker.update_step(config_id, "check_prerequisites", "completed" if prerequisites_ok else "failed")
                if not prerequisites_ok:
                    raise RuntimeError("check_prerequisites failed")

                tracker.set_current(config_id, "train_ner_model")
                tracker.update_step(config_id, "train_ner_model", "running")
                train_ok, model_dir_str = auto_pipeline.train_ner_model_with_config(config_id, config_info)
                model_dir = Path(model_dir_str)
                model_artifacts = collect_model_artifacts(model_dir)
                tracker.update_step(
                    config_id,
                    "train_ner_model",
                    "completed" if train_ok and model_artifacts["ready"] else "failed",
                    {"model_dir": str(model_dir), "model_artifacts": model_artifacts},
                )
                tracker.update_config(config_id, model_dir=str(model_dir), model_artifacts=model_artifacts)
                if not train_ok:
                    raise RuntimeError("train_ner_model_with_config failed")
                if not model_artifacts["ready"]:
                    raise RuntimeError(f"trained model artifacts incomplete for {config_id}: {model_artifacts}")

                tracker.set_current(config_id, "validate_model_artifacts")
                tracker.update_step(config_id, "validate_model_artifacts", "completed", model_artifacts)
                tracker.update_config(config_id, ner_metrics=model_artifacts)
                apply_model_env(model_dir)

                tracker.set_current(config_id, "extract_entity_words")
                tracker.update_step(config_id, "extract_entity_words", "running")
                extract_ok = auto_pipeline.extract_entity_words(config_id)
                tracker.update_step(config_id, "extract_entity_words", "completed" if extract_ok else "failed")
                if not extract_ok:
                    raise RuntimeError("extract_entity_words failed")

                tracker.set_current(config_id, "vectorize_and_store_to_es")
                tracker.update_step(config_id, "vectorize_and_store_to_es", "running")
                vectorize_ok = auto_pipeline.vectorize_and_store_to_es(config_id)
                tracker.update_step(config_id, "vectorize_and_store_to_es", "completed" if vectorize_ok else "failed")
                if not vectorize_ok:
                    raise RuntimeError("vectorize_and_store_to_es failed")

                tracker.set_current(config_id, "query_es_index_status")
                tracker.update_step(config_id, "query_es_index_status", "running")
                index_status = auto_pipeline.query_es_index_status(config_id)
                tracker.update_step(config_id, "query_es_index_status", "completed", {"document_count": index_status.get("document_count", 0)})
                tracker.update_config(config_id, es_index_status=index_status)

                tracker.set_current(config_id, "entity_linking_eval")
                eval_result = run_root_eval(
                    python_exe=args.python,
                    config_id=config_id,
                    model_dir=model_dir,
                    run_dir=run_dir,
                    tracker=tracker,
                    heartbeat_seconds=args.heartbeat_seconds,
                    limit=args.limit,
                    workers=args.workers,
                    llm_workers=args.llm_workers,
                    llm_retries=args.llm_retries,
                    llm_retry_delay=args.llm_retry_delay,
                    state_interval=args.state_interval,
                )
                tracker.update_config(
                    config_id,
                    search_metrics=eval_result.get("summary", {}),
                    eval_artifacts={
                        "summary_file": eval_result.get("summary_file"),
                        "eval_state_file": eval_result.get("eval_state_file"),
                        "stdout_log": eval_result.get("stdout_log"),
                        "stderr_log": eval_result.get("stderr_log"),
                    },
                    status="completed",
                    completed_at=now_text(),
                )
                tracker.update_step(config_id, "entity_linking_eval", "completed")
                for alias_config_id in alias_targets.get(config_id, []):
                    copy_config_result(
                        tracker,
                        config_id,
                        alias_config_id,
                        f"aliased to {config_id}; effective sources: {', '.join(plan_entry['effective_enabled_sources'])}",
                    )

            except Exception as exc:
                tracker.update_config(config_id, status="failed", error=str(exc), completed_at=now_text())
                tracker.update_step(config_id, tracker.state.get("current_step") or "unknown", "failed", {"error": str(exc)})
                for alias_config_id in alias_targets.get(config_id, []):
                    tracker.update_config(
                        alias_config_id,
                        status="blocked_by_alias_failure",
                        completed_at=now_text(),
                        error=f"canonical config {config_id} failed: {exc}",
                        note=f"canonical config {config_id} failed before aliasing",
                    )
                if args.stop_on_error:
                    raise
            finally:
                tracker.write()

        configs = tracker.state.get("configs", {})
        statuses = [(configs.get(config_id) or {}).get("status") for config_id in config_order]
        if any(status == "failed" for status in statuses):
            final_status = "completed_with_errors"
        elif any(status == "blocked_by_alias_failure" for status in statuses):
            final_status = "completed_with_blocked_aliases"
        elif any(status == "skipped_unavailable" for status in statuses):
            final_status = "completed_with_gaps"
        elif any(status in {"aliased", "pending_alias"} for status in statuses):
            final_status = "completed_with_aliases"
        else:
            final_status = "completed"
        tracker.update(status=final_status, finished_at=now_text())
        write_json(run_summary_path, deepcopy(tracker.state))
        tracker.write()
        return 0

    finally:
        tracker.set_current(None, None)
        try:
            shutil.copy2(config_backup_path, LOCAL_DIR / "config.py")
        except Exception:
            pass
        tracker.stop()


if __name__ == "__main__":
    raise SystemExit(main())
