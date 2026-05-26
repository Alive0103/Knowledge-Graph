from __future__ import annotations

import json
import logging
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import DEFAULT_DATASET
from .dbp15k import EMBEDDING_FILE_PREFIX
from .dbp15k import DBP15KDataset, KgKey
from .embedding_builder import (
    DEFAULT_BGE_M3_MODEL_DIR,
    DEFAULT_BGE_M3_MODEL_NAME,
    build_name_embedding_pickles,
)
from .evaluation import EvaluationResult, evaluate_alignment_model
from .model import ModelArgs, TORCH_AVAILABLE, create_alignment_model


try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    F = None
    DataLoader = None
    Dataset = object


logger = logging.getLogger(__name__)

DEFAULT_ALIGNMENT_TRAINING_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "alignment_training"
DEFAULT_LABSE_MODEL_NAME = "sentence-transformers/LaBSE"
DEFAULT_LABSE_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "alignment_baselines" / "labse"


@dataclass(frozen=True)
class AlignmentTrainingConfig:
    dataset_dir: Path
    output_dir: Path
    dataset_name: str = DEFAULT_DATASET
    device: str = "cpu"
    epochs: int = 150
    train_batch_size: int = 64
    eval_batch_size: int = 128
    queue_length: int = 64
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    temperature: float = 0.08
    momentum: float = 0.9999
    neighbor_size: int = 20
    dropout: float = 0.3
    gat_num: int = 1
    center_norm: bool = False
    neighbor_norm: bool = True
    emb_norm: bool = True
    combine: bool = True
    selection_metric: str = "valid_hits@1"
    seed: int = 37
    log_every_updates: int = 50
    max_train_updates_per_epoch: int | None = None
    embedding_name: str = "labse"
    embedding_model_name: str | None = None
    embedding_model_dir: Path | None = None
    labse_model_name: str = DEFAULT_LABSE_MODEL_NAME
    labse_model_dir: Path = DEFAULT_LABSE_MODEL_DIR
    hf_endpoint: str | None = None
    embedding_build_batch_size: int = 32
    embedding_build_max_length: int = 96

    def resolve_embedding_model_name(self) -> str:
        if self.embedding_model_name:
            return self.embedding_model_name
        if self.embedding_name == "bge_m3":
            return DEFAULT_BGE_M3_MODEL_NAME
        return self.labse_model_name

    def resolve_embedding_model_dir(self) -> Path:
        if self.embedding_model_dir is not None:
            return Path(self.embedding_model_dir)
        if self.embedding_name == "bge_m3":
            return DEFAULT_BGE_M3_MODEL_DIR
        return self.labse_model_dir

    def resolve_embedding_output_prefix(self) -> str:
        return EMBEDDING_FILE_PREFIX.get(self.embedding_name, self.embedding_name)

    def model_args(self, embedding_dim: int) -> ModelArgs:
        return ModelArgs(
            embedding_dim=int(embedding_dim),
            dropout=self.dropout,
            gat_num=self.gat_num,
            center_norm=self.center_norm,
            neighbor_norm=self.neighbor_norm,
            emb_norm=self.emb_norm,
            combine=self.combine,
            multi_head_dim=1,
        )

    def to_dict(self, embedding_dim: int | None = None) -> dict[str, object]:
        payload = asdict(self)
        payload["dataset_dir"] = str(self.dataset_dir)
        payload["output_dir"] = str(self.output_dir)
        payload["embedding_model_name"] = self.resolve_embedding_model_name()
        payload["embedding_model_dir"] = str(self.resolve_embedding_model_dir())
        payload["labse_model_dir"] = str(self.labse_model_dir)
        payload["model_args"] = asdict(self.model_args(embedding_dim=embedding_dim or 768))
        return payload


@dataclass(frozen=True)
class AlignmentEpochRecord:
    epoch: int
    train_loss: float
    update_count: int
    epoch_seconds: float
    valid_metrics: dict[str, object]
    test_metrics: dict[str, object]
    selection_metric: str
    selection_score: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AlignmentTrainingSummary:
    dataset: str
    output_dir: str
    best_model_path: str
    last_model_path: str
    summary_path: str
    history_path: str
    training_log_path: str
    selection_metric: str
    best_epoch: int
    best_score: float
    best_valid_metrics: dict[str, object]
    best_test_metrics: dict[str, object]
    raw_embedding_status: dict[str, object]
    training_config: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class NeighborGraphFeatureDataset(Dataset):
    def __init__(
        self,
        dataset: DBP15KDataset,
        kg: KgKey,
        neighbor_size: int,
        embedding_name: str = "labse",
    ) -> None:
        self.dataset = dataset
        self.kg = kg
        self.embedding_name = embedding_name
        self.neighbor_size = int(neighbor_size)
        self.embedding_dim = dataset.get_embedding_dim(kg, embedding_name=embedding_name)
        self.embeddings = dataset.load_raw_embeddings(kg, embedding_name=embedding_name)
        self.ordered_ids, self.neighbor_id_map = dataset.build_neighbor_ids(
            kg,
            neighbor_size=self.neighbor_size,
            embedding_name=embedding_name,
        )
        self._adj_cache = {
            valid_len: self._build_adj(valid_len)
            for valid_len in range(1, self.neighbor_size + 1)
        }

    def _build_adj(self, valid_len: int) -> np.ndarray:
        adj = np.zeros((self.neighbor_size, self.neighbor_size), dtype=np.float32)
        for pos in range(valid_len):
            adj[pos, pos] = 1.0
            adj[0, pos] = 1.0
            adj[pos, 0] = 1.0
        return adj

    def __len__(self) -> int:
        return len(self.ordered_ids)

    def __getitem__(self, index: int):
        entity_id = self.ordered_ids[index]
        current_neighbors = self.neighbor_id_map[entity_id]
        valid_len = min(len(current_neighbors), self.neighbor_size)

        entity_features = np.zeros((self.neighbor_size, self.embedding_dim), dtype=np.float32)
        for pos, neighbor_id in enumerate(current_neighbors[: self.neighbor_size]):
            entity_features[pos] = self.embeddings[neighbor_id]
        adj = self._adj_cache[valid_len]
        features = np.concatenate((entity_features, adj), axis=1)
        return torch.from_numpy(features), int(entity_id)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _momentum_update(teacher, student, momentum: float) -> None:
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)


def _extract_selection_score(result: EvaluationResult, metric: str) -> float:
    if metric == "valid_mrr":
        return float(result.mrr)
    if metric == "valid_hits@10":
        return float(result.hits_at.get(10, 0.0))
    return float(result.hits_at.get(1, 0.0))


def _save_checkpoint(
    path: Path,
    model,
    config: AlignmentTrainingConfig,
    epoch: int,
    valid_result: EvaluationResult,
    test_result: EvaluationResult,
    embedding_dim: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "training_config": config.to_dict(embedding_dim=embedding_dim),
        "valid_metrics": valid_result.to_dict(),
        "test_metrics": test_result.to_dict(),
    }
    torch.save(payload, path)


def _contrastive_queue_loss(
    query_vectors: "torch.Tensor",
    key_vectors: "torch.Tensor",
    negative_vectors: "torch.Tensor",
    temperature: float,
) -> "torch.Tensor":
    positive_logits = torch.sum(query_vectors * key_vectors, dim=1, keepdim=True)
    negative_logits = query_vectors @ negative_vectors.t()
    logits = torch.cat((positive_logits, negative_logits), dim=1) / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def ensure_raw_embeddings(
    dataset: DBP15KDataset,
    config: AlignmentTrainingConfig,
) -> dict[str, object]:
    output_prefix = config.resolve_embedding_output_prefix()
    left_path = dataset.dataset_dir / f"{output_prefix}_1.pkl"
    right_path = dataset.dataset_dir / f"{output_prefix}_2.pkl"
    if left_path.exists() and right_path.exists():
        return {
            "ready": True,
            "built": False,
            "embedding_name": config.embedding_name,
            "paths": {
                "1": str(left_path),
                "2": str(right_path),
            },
        }

    summary = build_name_embedding_pickles(
        dataset=dataset,
        model_name=config.resolve_embedding_model_name(),
        model_dir=config.resolve_embedding_model_dir(),
        embedding_name=config.embedding_name,
        output_prefix=output_prefix,
        overwrite=False,
        hf_endpoint=config.hf_endpoint,
        device=config.device,
        batch_size=config.embedding_build_batch_size,
        max_length=config.embedding_build_max_length,
    )
    return {
        "ready": True,
        "built": True,
        "embedding_name": config.embedding_name,
        "summary": summary.to_dict(),
    }


def train_alignment_model(config: AlignmentTrainingConfig) -> AlignmentTrainingSummary:
    if not TORCH_AVAILABLE:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for alignment model training")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    training_log_path = config.output_dir / "training.log"
    dataset = DBP15KDataset(dataset_dir=config.dataset_dir, dataset_name=config.dataset_name)
    raw_embedding_status = ensure_raw_embeddings(dataset, config)
    left_embedding_dim = dataset.get_embedding_dim("1", embedding_name=config.embedding_name)
    right_embedding_dim = dataset.get_embedding_dim("2", embedding_name=config.embedding_name)
    if left_embedding_dim != right_embedding_dim:
        raise RuntimeError(
            f"Embedding dim mismatch between KG1 ({left_embedding_dim}) and KG2 ({right_embedding_dim}) "
            f"for embedding_name={config.embedding_name}"
        )
    embedding_dim = int(left_embedding_dim)

    _set_seed(config.seed)
    file_handler = logging.FileHandler(training_log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    try:
        train_dataset_left = NeighborGraphFeatureDataset(
            dataset=dataset,
            kg="1",
            neighbor_size=config.neighbor_size,
            embedding_name=config.embedding_name,
        )
        train_dataset_right = NeighborGraphFeatureDataset(
            dataset=dataset,
            kg="2",
            neighbor_size=config.neighbor_size,
            embedding_name=config.embedding_name,
        )

        train_loader_left = DataLoader(
            train_dataset_left,
            batch_size=config.train_batch_size,
            shuffle=True,
            drop_last=True,
        )
        train_loader_right = DataLoader(
            train_dataset_right,
            batch_size=config.train_batch_size,
            shuffle=True,
            drop_last=True,
        )

        if len(train_loader_left) == 0 or len(train_loader_right) == 0:
            raise RuntimeError(
                "Training loaders are empty. Lower --train-batch-size or verify the DBP15K dataset."
            )

        model_args = config.model_args(embedding_dim=embedding_dim)
        student = create_alignment_model(device=config.device, args=model_args)
        teacher = create_alignment_model(device=config.device, args=model_args)
        teacher.load_state_dict(student.state_dict())
        teacher.eval()

        optimizer = torch.optim.Adam(
            student.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        best_model_path = config.output_dir / "best_model.pth"
        last_model_path = config.output_dir / "last_model.pth"
        history_path = config.output_dir / "history.json"
        summary_path = config.output_dir / "summary.json"

        history: list[dict[str, object]] = []
        best_epoch = -1
        best_score = float("-inf")
        best_valid_metrics: dict[str, object] | None = None
        best_test_metrics: dict[str, object] | None = None

        total_start = time.perf_counter()
        logger.info("Start alignment model training: dataset=%s output_dir=%s", config.dataset_name, config.output_dir)
        logger.info("Alignment raw embedding status: %s", json.dumps(raw_embedding_status, ensure_ascii=False))

        for epoch in range(1, config.epochs + 1):
            epoch_start = time.perf_counter()
            student.train()
            teacher.eval()

            queues: dict[str, deque] = {
                "1": deque(),
                "2": deque(),
            }
            iterators = {
                "1": iter(train_loader_left),
                "2": iter(train_loader_right),
            }
            available_kgs = ["1", "2"]
            rng = random.Random(config.seed + epoch)
            update_count = 0
            epoch_loss = 0.0

            while available_kgs:
                current_kg = rng.choice(available_kgs)
                try:
                    batch_features, _ = next(iterators[current_kg])
                except StopIteration:
                    available_kgs.remove(current_kg)
                    continue

                queues[current_kg].append(batch_features)
                if len(queues[current_kg]) < config.queue_length + 1:
                    continue

                positive_batch = queues[current_kg][0]
                negative_batches = list(queues[current_kg])[1:]
                queues[current_kg].popleft()

                positive_batch = positive_batch.to(config.device)
                negative_batch = torch.cat(negative_batches, dim=0).to(config.device)

                optimizer.zero_grad(set_to_none=True)
                query_vectors = student(positive_batch)
                with torch.no_grad():
                    key_vectors = teacher(positive_batch)
                    negative_vectors = teacher(negative_batch)
                loss = _contrastive_queue_loss(
                    query_vectors=query_vectors,
                    key_vectors=key_vectors,
                    negative_vectors=negative_vectors,
                    temperature=config.temperature,
                )
                loss.backward()
                if config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), config.gradient_clip_norm)
                optimizer.step()
                _momentum_update(teacher, student, config.momentum)

                update_count += 1
                epoch_loss += float(loss.item())

                if config.log_every_updates > 0 and update_count % config.log_every_updates == 0:
                    logger.info(
                        "epoch=%s update=%s loss=%.6f",
                        epoch,
                        update_count,
                        epoch_loss / update_count,
                    )

                if (
                    config.max_train_updates_per_epoch is not None
                    and update_count >= config.max_train_updates_per_epoch
                ):
                    logger.info(
                        "epoch=%s reached max_train_updates_per_epoch=%s, stop early for this epoch",
                        epoch,
                        config.max_train_updates_per_epoch,
                    )
                    break

            if update_count == 0:
                raise RuntimeError(
                    "Alignment training produced zero parameter updates. "
                    "Lower --queue-length or --train-batch-size."
                )

            train_loss = epoch_loss / update_count
            valid_result = evaluate_alignment_model(
                dataset=dataset,
                model=student,
                split="valid",
                batch_size=config.eval_batch_size,
                device=config.device,
                neighbor_size=config.neighbor_size,
                embedding_name=config.embedding_name,
                mode="trained_model_valid",
            )
            test_result = evaluate_alignment_model(
                dataset=dataset,
                model=student,
                split="test",
                batch_size=config.eval_batch_size,
                device=config.device,
                neighbor_size=config.neighbor_size,
                embedding_name=config.embedding_name,
                mode="trained_model_test",
            )
            selection_score = _extract_selection_score(valid_result, config.selection_metric)
            epoch_seconds = time.perf_counter() - epoch_start

            record = AlignmentEpochRecord(
                epoch=epoch,
                train_loss=round(train_loss, 6),
                update_count=update_count,
                epoch_seconds=round(epoch_seconds, 3),
                valid_metrics=valid_result.to_dict(),
                test_metrics=test_result.to_dict(),
                selection_metric=config.selection_metric,
                selection_score=round(selection_score, 6),
            )
            history.append(record.to_dict())
            history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

            _save_checkpoint(
                last_model_path,
                student,
                config,
                epoch=epoch,
                valid_result=valid_result,
                test_result=test_result,
                embedding_dim=embedding_dim,
            )

            logger.info(
                "epoch=%s train_loss=%.6f valid_mrr=%.3f valid_hits@1=%.3f test_mrr=%.3f test_hits@1=%.3f",
                epoch,
                train_loss,
                valid_result.mrr,
                valid_result.hits_at.get(1, 0.0),
                test_result.mrr,
                test_result.hits_at.get(1, 0.0),
            )

            if selection_score > best_score:
                best_score = selection_score
                best_epoch = epoch
                best_valid_metrics = valid_result.to_dict()
                best_test_metrics = test_result.to_dict()
                _save_checkpoint(
                    best_model_path,
                    student,
                    config,
                    epoch=epoch,
                    valid_result=valid_result,
                    test_result=test_result,
                    embedding_dim=embedding_dim,
                )
                logger.info(
                    "new best model saved: epoch=%s selection_metric=%s score=%.6f path=%s",
                    epoch,
                    config.selection_metric,
                    selection_score,
                    best_model_path,
                )

        total_seconds = round(time.perf_counter() - total_start, 3)
        if best_valid_metrics is None or best_test_metrics is None or best_epoch < 0:
            raise RuntimeError("Alignment training finished without producing a best checkpoint")

        summary = AlignmentTrainingSummary(
            dataset=config.dataset_name,
            output_dir=str(config.output_dir),
            best_model_path=str(best_model_path),
            last_model_path=str(last_model_path),
            summary_path=str(summary_path),
            history_path=str(history_path),
            training_log_path=str(training_log_path),
            selection_metric=config.selection_metric,
            best_epoch=best_epoch,
            best_score=round(best_score, 6),
            best_valid_metrics=best_valid_metrics,
            best_test_metrics=best_test_metrics,
            raw_embedding_status=raw_embedding_status,
            training_config={
                **config.to_dict(embedding_dim=embedding_dim),
                "total_training_seconds": total_seconds,
            },
        )
        summary_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Alignment training completed: %s", json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return summary
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()
