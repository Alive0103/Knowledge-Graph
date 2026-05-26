from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from model_hub import download_hf_snapshot
from vendor_utils import ensure_vendor_path

from .dbp15k import DBP15KDataset, KgKey


DEFAULT_BGE_M3_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_BGE_M3_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "alignment_baselines" / "bge-m3"
DEFAULT_BGE_M3_EMBEDDING_KEY = "bge_m3"
DEFAULT_BGE_M3_EMBEDDING_PREFIX = "raw_BGE_M3_emb"


@dataclass(frozen=True)
class EmbeddingBuildSummary:
    dataset: str
    embedding_name: str
    model_name: str
    model_dir: str
    device: str
    batch_size: int
    max_length: int
    embedding_dim: int
    output_files: dict[str, str]
    entity_counts: dict[str, int]
    metadata_file: str

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "embedding_name": self.embedding_name,
            "model_name": self.model_name,
            "model_dir": self.model_dir,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "embedding_dim": self.embedding_dim,
            "output_files": self.output_files,
            "entity_counts": self.entity_counts,
            "metadata_file": self.metadata_file,
        }


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def _prepare_text(text: str) -> str:
    normalized = str(text or "").strip()
    return normalized or "<EMPTY_ENTITY>"


def _load_transformer_components(model_dir: Path):
    ensure_vendor_path()

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir))
    return tokenizer, model


def _encode_texts(
    model_dir: Path,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    import torch

    tokenizer, model = _load_transformer_components(model_dir)
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()

    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device_obj) for key, value in encoded.items()}
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            batch_vectors = pooled.detach().cpu().numpy().astype(np.float32)
            vectors.append(batch_vectors)

    return _normalize_rows(np.concatenate(vectors, axis=0))


def build_name_embedding_pickles(
    dataset: DBP15KDataset,
    model_name: str = DEFAULT_BGE_M3_MODEL_NAME,
    model_dir: Path = DEFAULT_BGE_M3_MODEL_DIR,
    embedding_name: str = DEFAULT_BGE_M3_EMBEDDING_KEY,
    output_prefix: str = DEFAULT_BGE_M3_EMBEDDING_PREFIX,
    overwrite: bool = False,
    hf_endpoint: str | None = None,
    device: str = "cpu",
    batch_size: int = 32,
    max_length: int = 96,
) -> EmbeddingBuildSummary:
    model_dir = download_hf_snapshot(
        model_name=model_name,
        output_dir=Path(model_dir),
        overwrite=overwrite,
        hf_endpoint=hf_endpoint,
    )

    output_files: dict[str, str] = {}
    entity_counts: dict[str, int] = {}
    embedding_dim = 0

    for kg in ("1", "2"):
        output_path = dataset.dataset_dir / f"{output_prefix}_{kg}.pkl"
        if output_path.exists() and not overwrite:
            with open(output_path, "rb") as handle:
                payload = pickle.load(handle)
            first_vector = next(iter(payload.values()))
            first_array = np.asarray(first_vector, dtype=np.float32).reshape(-1)
            embedding_dim = int(first_array.shape[0])
            output_files[kg] = str(output_path)
            entity_counts[kg] = len(payload)
            continue

        ordered_ids = sorted(dataset.entities[kg].keys())
        texts = [_prepare_text(dataset.entities[kg][entity_id].name) for entity_id in ordered_ids]
        vectors = _encode_texts(
            model_dir=model_dir,
            texts=texts,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
        embedding_dim = int(vectors.shape[1])

        payload = {
            int(entity_id): vectors[idx]
            for idx, entity_id in enumerate(ordered_ids)
        }
        with open(output_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

        output_files[kg] = str(output_path)
        entity_counts[kg] = len(ordered_ids)

    metadata_path = dataset.dataset_dir / f"{output_prefix}_metadata.json"
    metadata = {
        "dataset": dataset.dataset_name,
        "embedding_name": embedding_name,
        "model_name": model_name,
        "model_dir": str(model_dir),
        "device": device,
        "batch_size": batch_size,
        "max_length": max_length,
        "embedding_dim": embedding_dim,
        "output_prefix": output_prefix,
        "output_files": output_files,
        "entity_counts": entity_counts,
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return EmbeddingBuildSummary(
        dataset=dataset.dataset_name,
        embedding_name=embedding_name,
        model_name=model_name,
        model_dir=str(model_dir),
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        embedding_dim=embedding_dim,
        output_files=output_files,
        entity_counts=entity_counts,
        metadata_file=str(metadata_path),
    )
