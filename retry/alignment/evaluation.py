from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .dbp15k import DBP15KDataset, SplitKey
from .model import ModelArgs, TORCH_AVAILABLE, create_alignment_model, load_checkpoint


@dataclass(frozen=True)
class EvaluationResult:
    dataset: str
    mode: str
    split: str
    query_count: int
    candidate_count: int
    hits_at: dict[int, float]
    mrr: float
    model_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "dataset": self.dataset,
            "mode": self.mode,
            "split": self.split,
            "query_count": self.query_count,
            "candidate_count": self.candidate_count,
            "mrr": self.mrr,
        }
        for k, value in sorted(self.hits_at.items()):
            payload[f"hits@{k}"] = value
        if self.model_path is not None:
            payload["model_path"] = self.model_path
        return payload


def _coerce_top_k(top_k: Sequence[int] | int) -> tuple[int, ...]:
    if isinstance(top_k, int):
        return (top_k,)
    return tuple(sorted({int(value) for value in top_k if int(value) > 0}))


def _build_eval_inputs(
    dataset: DBP15KDataset,
    split: SplitKey,
    left_vectors: np.ndarray,
    left_ids: np.ndarray,
    right_vectors: np.ndarray,
    right_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left_index = {int(entity_id): idx for idx, entity_id in enumerate(left_ids.tolist())}
    right_index = {int(entity_id): idx for idx, entity_id in enumerate(right_ids.tolist())}

    query_indices: list[int] = []
    target_indices: list[int] = []
    for pair in dataset.get_alignment_pairs(split):
        left_idx = left_index.get(pair.left_id)
        right_idx = right_index.get(pair.right_id)
        if left_idx is None or right_idx is None:
            continue
        query_indices.append(left_idx)
        target_indices.append(right_idx)

    if not query_indices:
        raise RuntimeError(f"No valid alignment pairs found for split={split}")

    query_matrix = left_vectors[np.asarray(query_indices, dtype=np.int64)]
    target_matrix = np.asarray(target_indices, dtype=np.int64)
    return query_matrix, target_matrix


def _compute_metrics(
    query_matrix: np.ndarray,
    candidate_matrix: np.ndarray,
    target_indices: np.ndarray,
    top_k: tuple[int, ...],
    batch_size: int = 256,
) -> tuple[dict[int, float], float]:
    candidate_positions = np.arange(candidate_matrix.shape[0], dtype=np.int64)
    hit_counts = {k: 0 for k in top_k}
    reciprocal_rank_sum = 0.0

    for start in range(0, query_matrix.shape[0], batch_size):
        end = min(start + batch_size, query_matrix.shape[0])
        batch_queries = query_matrix[start:end]
        batch_targets = target_indices[start:end]

        scores = batch_queries @ candidate_matrix.T
        target_scores = scores[np.arange(scores.shape[0]), batch_targets]
        higher = (scores > target_scores[:, None]).sum(axis=1)
        tied_before = (
            (scores == target_scores[:, None])
            & (candidate_positions[None, :] < batch_targets[:, None])
        ).sum(axis=1)
        ranks = 1 + higher + tied_before

        reciprocal_rank_sum += np.sum(1.0 / ranks)
        for k in top_k:
            hit_counts[k] += int(np.sum(ranks <= k))

    total = int(query_matrix.shape[0])
    hits_at = {k: round(hit_counts[k] / total, 3) for k in top_k}
    mrr = round(reciprocal_rank_sum / total, 3)
    return hits_at, mrr


def evaluate_raw_alignment(
    dataset: DBP15KDataset,
    split: SplitKey = "test",
    top_k: Sequence[int] | int = (1, 5, 10),
    batch_size: int = 256,
    embedding_name: str = "labse",
) -> EvaluationResult:
    top_k = _coerce_top_k(top_k)
    left_ids, left_vectors = dataset.get_embedding_matrix("1", embedding_name=embedding_name)
    right_ids, right_vectors = dataset.get_embedding_matrix("2", embedding_name=embedding_name)
    query_matrix, target_indices = _build_eval_inputs(dataset, split, left_vectors, left_ids, right_vectors, right_ids)
    hits_at, mrr = _compute_metrics(query_matrix, right_vectors, target_indices, top_k, batch_size=batch_size)
    return EvaluationResult(
        dataset=dataset.dataset_name,
        mode="raw" if embedding_name == "labse" else f"raw_{embedding_name}",
        split=split,
        query_count=int(query_matrix.shape[0]),
        candidate_count=int(right_vectors.shape[0]),
        hits_at=hits_at,
        mrr=mrr,
    )


def _iter_model_feature_batches(
    dataset: DBP15KDataset,
    kg: str,
    batch_size: int,
    neighbor_size: int,
    embedding_name: str = "labse",
):
    embeddings = dataset.load_raw_embeddings(kg, embedding_name=embedding_name)
    embedding_dim = dataset.get_embedding_dim(kg, embedding_name=embedding_name)
    ordered_ids, neighbor_id_map = dataset.build_neighbor_ids(kg, neighbor_size=neighbor_size, embedding_name=embedding_name)

    for start in range(0, len(ordered_ids), batch_size):
        batch_ids = ordered_ids[start:start + batch_size]
        batch_features = np.zeros((len(batch_ids), neighbor_size, embedding_dim + neighbor_size), dtype=np.float32)
        for row_idx, entity_id in enumerate(batch_ids):
            current_neighbors = neighbor_id_map[entity_id]
            valid_len = min(len(current_neighbors), neighbor_size)
            entity_features = np.zeros((neighbor_size, embedding_dim), dtype=np.float32)
            adj = np.zeros((neighbor_size, neighbor_size), dtype=np.float32)
            for pos, neighbor_id in enumerate(current_neighbors[:neighbor_size]):
                entity_features[pos] = embeddings[neighbor_id]
            for pos in range(valid_len):
                adj[pos, pos] = 1.0
                adj[0, pos] = 1.0
                adj[pos, 0] = 1.0
            batch_features[row_idx] = np.concatenate((entity_features, adj), axis=1)
        yield batch_ids, batch_features


def _encode_with_final_model(
    dataset: DBP15KDataset,
    kg: str,
    model_path: str | Path,
    batch_size: int,
    device: str,
    neighbor_size: int,
    embedding_name: str = "labse",
) -> tuple[np.ndarray, np.ndarray]:
    if not TORCH_AVAILABLE:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for final_model evaluation")

    import torch

    model = create_alignment_model(device=device)
    load_checkpoint(model, model_path=model_path, device=device)

    collected_ids: list[int] = []
    collected_vectors: list[np.ndarray] = []
    with torch.no_grad():
        for batch_ids, batch_features in _iter_model_feature_batches(
            dataset=dataset,
            kg=kg,
            batch_size=batch_size,
            neighbor_size=neighbor_size,
            embedding_name=embedding_name,
        ):
            tensor = torch.from_numpy(batch_features).to(device)
            encoded = model(tensor).detach().cpu().numpy().astype(np.float32)
            norms = np.linalg.norm(encoded, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            encoded = encoded / norms
            collected_ids.extend(batch_ids)
            collected_vectors.append(encoded)

    return np.asarray(collected_ids, dtype=np.int64), np.concatenate(collected_vectors, axis=0)


def encode_alignment_model(
    dataset: DBP15KDataset,
    kg: str,
    model,
    batch_size: int = 128,
    device: str = "cpu",
    neighbor_size: int = 20,
    embedding_name: str = "labse",
) -> tuple[np.ndarray, np.ndarray]:
    if not TORCH_AVAILABLE:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for graph-model evaluation")

    import torch

    model = model.to(device)
    model.eval()

    collected_ids: list[int] = []
    collected_vectors: list[np.ndarray] = []
    with torch.no_grad():
        for batch_ids, batch_features in _iter_model_feature_batches(
            dataset=dataset,
            kg=kg,
            batch_size=batch_size,
            neighbor_size=neighbor_size,
            embedding_name=embedding_name,
        ):
            tensor = torch.from_numpy(batch_features).to(device)
            encoded = model(tensor).detach().cpu().numpy().astype(np.float32)
            norms = np.linalg.norm(encoded, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            encoded = encoded / norms
            collected_ids.extend(batch_ids)
            collected_vectors.append(encoded)

    return np.asarray(collected_ids, dtype=np.int64), np.concatenate(collected_vectors, axis=0)


def evaluate_alignment_model(
    dataset: DBP15KDataset,
    model,
    split: SplitKey = "test",
    top_k: Sequence[int] | int = (1, 5, 10),
    batch_size: int = 128,
    device: str = "cpu",
    neighbor_size: int = 20,
    embedding_name: str = "labse",
    mode: str = "trained_model",
    model_path: str | None = None,
) -> EvaluationResult:
    top_k = _coerce_top_k(top_k)
    left_ids, left_vectors = encode_alignment_model(
        dataset=dataset,
        kg="1",
        model=model,
        batch_size=batch_size,
        device=device,
        neighbor_size=neighbor_size,
        embedding_name=embedding_name,
    )
    right_ids, right_vectors = encode_alignment_model(
        dataset=dataset,
        kg="2",
        model=model,
        batch_size=batch_size,
        device=device,
        neighbor_size=neighbor_size,
        embedding_name=embedding_name,
    )
    query_matrix, target_indices = _build_eval_inputs(dataset, split, left_vectors, left_ids, right_vectors, right_ids)
    hits_at, mrr = _compute_metrics(query_matrix, right_vectors, target_indices, top_k, batch_size=batch_size)
    return EvaluationResult(
        dataset=dataset.dataset_name,
        mode=mode,
        split=split,
        query_count=int(query_matrix.shape[0]),
        candidate_count=int(right_vectors.shape[0]),
        hits_at=hits_at,
        mrr=mrr,
        model_path=model_path,
    )


def evaluate_final_model_alignment(
    dataset: DBP15KDataset,
    model_path: str | Path,
    split: SplitKey = "test",
    top_k: Sequence[int] | int = (1, 5, 10),
    batch_size: int = 128,
    device: str = "cpu",
    neighbor_size: int = 20,
    embedding_name: str = "labse",
) -> EvaluationResult:
    embedding_dim = dataset.get_embedding_dim("1", embedding_name=embedding_name)
    model = create_alignment_model(
        device=device,
        args=ModelArgs(embedding_dim=embedding_dim),
    )
    load_checkpoint(model, model_path=model_path, device=device)
    return evaluate_alignment_model(
        dataset=dataset,
        model=model,
        split=split,
        top_k=top_k,
        batch_size=batch_size,
        device=device,
        neighbor_size=neighbor_size,
        embedding_name=embedding_name,
        mode="final_model",
        model_path=str(Path(model_path)),
    )
