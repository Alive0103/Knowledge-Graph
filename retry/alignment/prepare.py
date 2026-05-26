from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import DEFAULT_DATASET, REPO_ROOT, REQUIRED_DBP15K_FILES


@dataclass(frozen=True)
class AlignmentAssetSummary:
    dataset: str
    target_root: str
    target_dataset_dir: str
    source_kind: str
    source_path: str
    restored_files: list[str]
    copied_embedding_files: list[str]
    is_complete: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _dataset_is_complete(dataset_dir: Path) -> bool:
    return dataset_dir.exists() and all((dataset_dir / name).exists() for name in REQUIRED_DBP15K_FILES)


def _copy_text_file(source: Path, target: Path) -> None:
    text = source.read_text(encoding="utf-8")
    target.write_text(text, encoding="utf-8")


def _restore_from_git_head(repo_root: Path, git_path_prefix: str, target_dir: Path) -> list[str]:
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        return []

    restored: list[str] = []
    for name in REQUIRED_DBP15K_FILES:
        git_object = f"HEAD:{git_path_prefix}/{name}"
        result = subprocess.run(
            ["git", "show", git_object],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            continue
        (target_dir / name).write_bytes(result.stdout)
        restored.append(name)
    return restored


def _copy_embedding_files(source_dir: Path, target_dir: Path) -> list[str]:
    copied: list[str] = []
    for name in ("raw_LaBSE_emb_1.pkl", "raw_LaBSE_emb_2.pkl", "raw_BGE_M3_emb_1.pkl", "raw_BGE_M3_emb_2.pkl"):
        source = source_dir / name
        if not source.exists():
            continue
        target = target_dir / name
        if target.exists():
            continue
        target.write_bytes(source.read_bytes())
        copied.append(name)
    return copied


def prepare_alignment_dataset(
    dataset: str = DEFAULT_DATASET,
    repo_root: Path = REPO_ROOT,
    target_root: Path | None = None,
) -> AlignmentAssetSummary:
    repo_root = Path(repo_root)
    target_root = Path(target_root) if target_root is not None else repo_root / "retry" / "recovered" / "alignment" / "DBP15K"
    target_dataset_dir = target_root / dataset
    target_dataset_dir.mkdir(parents=True, exist_ok=True)

    direct_candidates: list[tuple[str, Path]] = [
        ("primary_processed", repo_root / "data" / "processed" / "alignment" / "DBP15K" / dataset),
        ("knowledge_server_processed", repo_root / "knowledge_server" / "data" / "processed" / "alignment" / "DBP15K" / dataset),
        ("archived_selfkg_main", repo_root / "archived" / "SelfKG-original" / "SelfKG-main" / "data" / "DBP15K" / dataset),
        ("archived_dq", repo_root / "archived" / "SelfKG-original" / "dq" / "data" / "DBP15K" / dataset),
    ]

    for source_kind, source_dir in direct_candidates:
        if _dataset_is_complete(source_dir):
            restored: list[str] = []
            for name in REQUIRED_DBP15K_FILES:
                target = target_dataset_dir / name
                if target.exists():
                    continue
                _copy_text_file(source_dir / name, target)
                restored.append(name)
            copied_embedding_files = _copy_embedding_files(source_dir, target_dataset_dir)
            return AlignmentAssetSummary(
                dataset=dataset,
                target_root=str(target_root),
                target_dataset_dir=str(target_dataset_dir),
                source_kind=source_kind,
                source_path=str(source_dir),
                restored_files=restored,
                copied_embedding_files=copied_embedding_files,
                is_complete=_dataset_is_complete(target_dataset_dir),
            )

    restored_from_git = _restore_from_git_head(
        repo_root=repo_root,
        git_path_prefix=f"data/processed/alignment/DBP15K/{dataset}",
        target_dir=target_dataset_dir,
    )

    copied_embedding_files: list[str] = []
    for _, source_dir in direct_candidates:
        copied_embedding_files = _copy_embedding_files(source_dir, target_dataset_dir)
        if copied_embedding_files:
            break

    return AlignmentAssetSummary(
        dataset=dataset,
        target_root=str(target_root),
        target_dataset_dir=str(target_dataset_dir),
        source_kind="git_head" if restored_from_git else "none",
        source_path=f"HEAD:data/processed/alignment/DBP15K/{dataset}" if restored_from_git else "",
        restored_files=restored_from_git,
        copied_embedding_files=copied_embedding_files,
        is_complete=_dataset_is_complete(target_dataset_dir),
    )
