from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alignment.config import DEFAULT_DATASET, DEFAULT_DBP15K_ROOT
from entity_linking.config import REPO_ROOT
from entity_linking.es_index import DEFAULT_ES_INDEX_NAME, DEFAULT_ES_URL


def _choose_existing(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _latest_model_path(prefix: str) -> Path | None:
    root = REPO_ROOT / "retry" / "output" / "alignment_training"
    if not root.exists():
        return None
    matches = sorted(root.glob(f"{prefix}*/best_model.pth"))
    if not matches:
        return None
    return matches[-1]


def default_entity_linking_processed_dir() -> Path:
    return _choose_existing(
        REPO_ROOT
        / "retry"
        / "output"
        / "entity_linking_transformer_distilbert_mbert_rigorous_full_overnight_complete_20260331_001_labse",
        REPO_ROOT / "retry" / "output" / "entity_linking",
    ) or (REPO_ROOT / "retry" / "output" / "entity_linking")


def default_alignment_model_path() -> Path | None:
    return _choose_existing(
        REPO_ROOT
        / "retry"
        / "output"
        / "alignment_training"
        / "labse_neighbor_retrained_zh_en_rigorous_full_overnight_complete_20260331_001_labse"
        / "best_model.pth",
        _latest_model_path("labse_neighbor_retrained_zh_en"),
    )


def default_bge_alignment_model_path() -> Path | None:
    return _choose_existing(
        REPO_ROOT
        / "retry"
        / "output"
        / "alignment_training"
        / "bge_m3_neighbor_retrained_zh_en_overnight_complete_20260331_001_bge_graph"
        / "best_model.pth",
        _latest_model_path("bge_m3_neighbor_retrained_zh_en"),
    )


@dataclass(frozen=True)
class KgRagConfig:
    kg_repo_root: Path = REPO_ROOT
    dbp15k_root: Path = DEFAULT_DBP15K_ROOT
    dbp15k_dataset: str = DEFAULT_DATASET
    entity_linking_processed_dir: Path = default_entity_linking_processed_dir()
    alignment_model_path: Path | None = default_alignment_model_path()
    bge_alignment_model_path: Path | None = default_bge_alignment_model_path()
    embedding_family: str = "labse"
    es_url: str = DEFAULT_ES_URL
    es_index_name: str = DEFAULT_ES_INDEX_NAME
    enable_alignment_expansion: bool = True
    default_query_intent: str = "auto"
    default_retrieval_mode: str = "hybrid"
    default_top_k: int = 10
    default_relation_limit: int = 10
    default_triple_limit: int = 10
    default_kg_side: str = "auto"

    @property
    def dataset_dir(self) -> Path:
        return self.dbp15k_root / self.dbp15k_dataset

    def validate_required_paths(self) -> dict[str, str]:
        issues: dict[str, str] = {}
        if not self.kg_repo_root.exists():
            issues["kg_repo_root"] = str(self.kg_repo_root)
        if not self.dbp15k_root.exists():
            issues["dbp15k_root"] = str(self.dbp15k_root)
        if not self.dataset_dir.exists():
            issues["dataset_dir"] = str(self.dataset_dir)
        if not self.entity_linking_processed_dir.exists():
            issues["entity_linking_processed_dir"] = str(self.entity_linking_processed_dir)
        if self.alignment_model_path is None or not Path(self.alignment_model_path).exists():
            issues["alignment_model_path"] = str(self.alignment_model_path or "")
        return issues

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "KgRagConfig":
        payload = dict(payload or {})
        defaults = cls()
        values = {
            "kg_repo_root": Path(payload.get("kg_repo_root") or os.getenv("KG_RAG_REPO_ROOT") or defaults.kg_repo_root),
            "dbp15k_root": Path(
                payload.get("dbp15k_root") or os.getenv("KG_RAG_DBP15K_ROOT") or defaults.dbp15k_root
            ),
            "dbp15k_dataset": str(
                payload.get("dbp15k_dataset") or os.getenv("KG_RAG_DEFAULT_DATASET") or defaults.dbp15k_dataset
            ),
            "entity_linking_processed_dir": Path(
                payload.get("entity_linking_processed_dir") or defaults.entity_linking_processed_dir
            ),
            "alignment_model_path": Path(payload["alignment_model_path"])
            if payload.get("alignment_model_path")
            else defaults.alignment_model_path,
            "bge_alignment_model_path": Path(payload["bge_alignment_model_path"])
            if payload.get("bge_alignment_model_path")
            else defaults.bge_alignment_model_path,
            "embedding_family": str(payload.get("embedding_family") or defaults.embedding_family),
            "es_url": str(payload.get("es_url") or os.getenv("KG_RAG_DEFAULT_ES_URL") or defaults.es_url),
            "es_index_name": str(
                payload.get("es_index_name") or os.getenv("KG_RAG_DEFAULT_ES_INDEX") or defaults.es_index_name
            ),
            "enable_alignment_expansion": bool(
                payload.get("enable_alignment_expansion", defaults.enable_alignment_expansion)
            ),
            "default_query_intent": str(payload.get("default_query_intent") or defaults.default_query_intent),
            "default_retrieval_mode": str(payload.get("default_retrieval_mode") or defaults.default_retrieval_mode),
            "default_top_k": int(payload.get("default_top_k", defaults.default_top_k)),
            "default_relation_limit": int(payload.get("default_relation_limit", defaults.default_relation_limit)),
            "default_triple_limit": int(payload.get("default_triple_limit", defaults.default_triple_limit)),
            "default_kg_side": str(payload.get("default_kg_side") or defaults.default_kg_side),
        }
        return cls(**values)
