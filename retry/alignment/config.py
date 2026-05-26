from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() or (candidate / "data").exists():
            return candidate
    return start.resolve().parents[2]


REPO_ROOT = find_repo_root(Path(__file__).resolve())
DEFAULT_DATASET = "zh_en"
REQUIRED_DBP15K_FILES = (
    "cleaned_ent_ids_1",
    "cleaned_ent_ids_2",
    "cleaned_rel_ids_1",
    "cleaned_rel_ids_2",
    "ref_ent_ids",
    "test",
    "triples_1",
    "triples_2",
    "valid",
)


def has_complete_dbp15k_dataset(root: Path, dataset: str = DEFAULT_DATASET) -> bool:
    dataset_dir = Path(root) / dataset
    return dataset_dir.exists() and all((dataset_dir / name).exists() for name in REQUIRED_DBP15K_FILES)


def choose_existing_dbp15k_root(*candidates: Path, dataset: str = DEFAULT_DATASET) -> Path:
    for candidate in candidates:
        if has_complete_dbp15k_dataset(candidate, dataset=dataset):
            return candidate
    return Path(candidates[0])


DEFAULT_DBP15K_ROOT = choose_existing_dbp15k_root(
    REPO_ROOT / "data" / "processed" / "alignment" / "DBP15K",
    REPO_ROOT / "retry" / "recovered" / "alignment" / "DBP15K",
    REPO_ROOT / "knowledge_server" / "data" / "processed" / "alignment" / "DBP15K",
)


def choose_existing(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


DEFAULT_MODEL_PATH = choose_existing(
    REPO_ROOT / "data" / "models" / "final_model.pth",
    REPO_ROOT / "跨语言实体对齐" / "final_model.pth",
)


@dataclass(frozen=True)
class AlignmentRuntimeConfig:
    dbp15k_root: Path = DEFAULT_DBP15K_ROOT
    dataset: str = DEFAULT_DATASET
    model_path: Path | None = DEFAULT_MODEL_PATH
    neighbor_size: int = 20
    embedding_dim: int = 768
    eval_batch_size: int = 128
    top_k: tuple[int, ...] = (1, 5, 10)

    @property
    def dataset_dir(self) -> Path:
        return self.dbp15k_root / self.dataset
