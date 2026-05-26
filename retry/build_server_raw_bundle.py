#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


RETRY_DIR = Path(__file__).resolve().parent
REPO_ROOT = RETRY_DIR.parent
if str(RETRY_DIR) not in sys.path:
    sys.path.insert(0, str(RETRY_DIR))

from alignment.config import REQUIRED_DBP15K_FILES
from alignment.prepare import prepare_alignment_dataset


DEFAULT_BUNDLE_DIR = REPO_ROOT / "server_upload_raw_only"
DEFAULT_DATASET = "zh_en"

RETRY_TOP_LEVEL_FILES = (
    "README.md",
    "SERVER_RUN_GUIDE.md",
    "OVERNIGHT_RECOVERY_GUIDE.md",
    "requirements_server.txt",
    "build_server_raw_bundle.py",
    "model_hub.py",
    "vendor_utils.py",
    "run_alignment.py",
    "run_alignment_embedding_baseline.py",
    "run_entity_linking.py",
    "run_entity_linking_es.py",
    "run_entity_linking_training.py",
    "run_experiment_comparison.py",
    "run_full_pipeline.py",
    "run_overnight_recovery.py",
    "run_prepare_experiment_assets.py",
    "run_rigorous_full_experiment.py",
)

RAW_INPUT_FILES = (
    ("work_wyy/data/zh_wiki_v2.jsonl", "work_wyy/data/zh_wiki_v2.jsonl"),
    ("work_wyy/data/en_wiki_v3.jsonl", "work_wyy/data/en_wiki_v3.jsonl"),
    ("work_wyy/data/find.xlsx", "work_wyy/data/find.xlsx"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a raw-only server upload bundle without models or intermediate artifacts")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    return parser


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def bundle_readme(dataset: str) -> str:
    return f"""# Raw-Only Server Upload Bundle

这个目录是给新服务器上传用的最小实验包，只保留：

- 原始 wiki 输入数据
- DBP15K `{dataset}` 的文本数据文件
- `retry/` 必要源码

这个目录明确不包含：

- `retry/output/` 下的任何中间产物
- `retry/models/` 下的任何基础模型或下载模型
- `raw_LaBSE_emb_*.pkl`
- `raw_BGE_M3_emb_*.pkl`
- `final_model.pth`
- `retry/_vendor/`

上传到服务器后，建议在这个目录根部执行：

```bash
python retry/run_prepare_experiment_assets.py --dataset {dataset} --check-es --prepare-bge-model --json
python retry/run_rigorous_full_experiment.py --dataset {dataset} --device cuda:0 --include-bge-m3
```

注意：

- 由于本包不包含 `final_model.pth`，如果服务器上也没有该权重，则严格流程会跳过 `alignment_final_model`
- 由于本包不包含 `raw_LaBSE_emb_*.pkl`，严格流程会在缺失时自动下载 `LaBSE` 并现生成 raw baseline
- 如果你只想先验环境，可以先跑 `retry/run_prepare_experiment_assets.py`
"""


def build_manifest(dataset: str, output_dir: Path, copied_files: list[str]) -> dict[str, object]:
    return {
        "bundle_dir": str(output_dir),
        "dataset": dataset,
        "copied_files": copied_files,
        "excluded_artifacts": [
            "retry/output/**",
            "retry/models/**",
            "retry/_vendor/**",
            "data/models/final_model.pth",
            f"data/processed/alignment/DBP15K/{dataset}/raw_LaBSE_emb_*.pkl",
            f"data/processed/alignment/DBP15K/{dataset}/raw_BGE_M3_emb_*.pkl",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []

    for relative_path in RETRY_TOP_LEVEL_FILES:
        source = RETRY_DIR / relative_path
        target = output_dir / "retry" / relative_path
        copy_file(source, target)
        copied_files.append(str(target.relative_to(output_dir)).replace("\\", "/"))

    for package_name in ("alignment", "entity_linking"):
        package_dir = RETRY_DIR / package_name
        for source in sorted(package_dir.iterdir()):
            if not source.is_file():
                continue
            if source.suffix not in {".py", ".md"}:
                continue
            target = output_dir / "retry" / package_name / source.name
            copy_file(source, target)
            copied_files.append(str(target.relative_to(output_dir)).replace("\\", "/"))

    for source_relative, target_relative in RAW_INPUT_FILES:
        source = REPO_ROOT / source_relative
        target = output_dir / target_relative
        copy_file(source, target)
        copied_files.append(str(target.relative_to(output_dir)).replace("\\", "/"))

    alignment_summary = prepare_alignment_dataset(dataset=args.dataset, repo_root=REPO_ROOT)
    alignment_source_dir = Path(alignment_summary.target_dataset_dir)
    alignment_target_dir = output_dir / "data" / "processed" / "alignment" / "DBP15K" / args.dataset
    for name in REQUIRED_DBP15K_FILES:
        source = alignment_source_dir / name
        target = alignment_target_dir / name
        copy_file(source, target)
        copied_files.append(str(target.relative_to(output_dir)).replace("\\", "/"))

    write_text(output_dir / "README.md", bundle_readme(args.dataset))
    copied_files.append("README.md")

    manifest_path = output_dir / "bundle_manifest.json"
    write_text(manifest_path, json.dumps(build_manifest(args.dataset, output_dir, copied_files), ensure_ascii=False, indent=2))
    copied_files.append("bundle_manifest.json")

    print(
        json.dumps(
            {
                "bundle_dir": str(output_dir),
                "dataset": args.dataset,
                "copied_file_count": len(copied_files),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
