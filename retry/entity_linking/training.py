from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from vendor_utils import ensure_vendor_path

ensure_vendor_path()

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

from entity_linking.io_utils import iter_jsonl


logger = logging.getLogger(__name__)

DEFAULT_LABEL_TO_ID = {"O": 0, "B-ENTITY": 1, "I-ENTITY": 2}
DEFAULT_ID_TO_LABEL = {value: key for key, value in DEFAULT_LABEL_TO_ID.items()}


@dataclass(frozen=True)
class NERTrainingConfig:
    model_name_or_path: str
    train_path: Path
    dev_path: Path
    output_dir: Path
    max_length: int = 192
    batch_size: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    epochs: int = 1
    gradient_accumulation_steps: int = 1
    device: str = "cpu"
    seed: int = 42
    max_train_examples: int | None = None
    max_dev_examples: int | None = None
    log_every_steps: int = 10


@dataclass
class TrainingMetrics:
    loss: float
    token_accuracy: float
    positive_precision: float
    positive_recall: float
    positive_f1: float


@dataclass
class TrainingSummary:
    model_name_or_path: str
    train_examples: int
    dev_examples: int
    epochs: int
    batch_size: int
    max_length: int
    device: str
    num_labels: int
    label_names: list[str]
    best_epoch: int
    best_dev_positive_f1: float
    best_dev_metrics: dict[str, float]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_examples(path: Path, max_examples: int | None = None) -> list[dict]:
    return list(iter_jsonl(path, limit=max_examples))


def _normalize_entity_type(raw_value: object) -> str:
    candidate = str(raw_value or "").strip()
    return candidate or "ENTITY"


def build_label_mapping(examples: list[dict]) -> tuple[dict[str, int], dict[int, str]]:
    entity_types: set[str] = set()
    for example in examples:
        raw_labels = example.get("labels")
        if isinstance(raw_labels, list) and raw_labels:
            for raw_label in raw_labels:
                label = str(raw_label or "O").strip()
                if not label or label == "O":
                    continue
                if label.startswith(("B-", "I-")) and len(label) > 2:
                    entity_types.add(label[2:])
                else:
                    entity_types.add(label)
            continue

        for entity in list(example.get("entities") or []):
            entity_types.add(_normalize_entity_type(entity.get("label") or entity.get("type")))

    if not entity_types:
        return dict(DEFAULT_LABEL_TO_ID), dict(DEFAULT_ID_TO_LABEL)

    label_to_id = {"O": 0}
    next_id = 1
    for entity_type in sorted(entity_types):
        label_to_id[f"B-{entity_type}"] = next_id
        next_id += 1
        label_to_id[f"I-{entity_type}"] = next_id
        next_id += 1
    id_to_label = {value: key for key, value in label_to_id.items()}
    return label_to_id, id_to_label


def _build_char_labels_from_entities(text: str, entities: list[dict], label_to_id: dict[str, int]) -> list[int]:
    labels = [label_to_id["O"]] * len(text)
    for entity in sorted(entities, key=lambda item: (int(item["start"]), int(item["end"]))):
        start = max(0, min(len(text), int(entity["start"])))
        end = max(start, min(len(text), int(entity["end"])))
        if end <= start:
            continue
        entity_type = _normalize_entity_type(entity.get("label") or entity.get("type"))
        begin_label = label_to_id.get(f"B-{entity_type}", label_to_id["O"])
        inside_label = label_to_id.get(f"I-{entity_type}", begin_label)
        labels[start] = begin_label
        for index in range(start + 1, end):
            labels[index] = inside_label
    return labels


def _build_char_labels(example: dict, text: str, label_to_id: dict[str, int]) -> list[int]:
    raw_labels = example.get("labels")
    if isinstance(raw_labels, list) and raw_labels:
        normalized_labels = [label_to_id["O"]] * len(text)
        limit = min(len(text), len(raw_labels))
        for index in range(limit):
            label_name = str(raw_labels[index] or "O").strip() or "O"
            normalized_labels[index] = label_to_id.get(label_name, label_to_id["O"])
        return normalized_labels
    return _build_char_labels_from_entities(text, list(example.get("entities") or []), label_to_id)


def _labels_from_offsets(offsets, char_labels: list[int], outside_label_id: int) -> list[int]:
    token_labels: list[int] = []
    for start, end in offsets:
        start = int(start)
        end = int(end)
        if start == end:
            token_labels.append(-100)
            continue
        if start >= len(char_labels):
            token_labels.append(outside_label_id)
            continue

        span_labels = char_labels[start:end]
        if not span_labels or max(span_labels) == outside_label_id:
            token_labels.append(outside_label_id)
        elif char_labels[start] != outside_label_id:
            token_labels.append(char_labels[start])
        else:
            non_zero = next((value for value in span_labels if value != outside_label_id), outside_label_id)
            token_labels.append(non_zero)
    return token_labels


class TokenClassificationDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_length: int, label_to_id: dict[str, int]) -> None:
        self.items: list[dict[str, torch.Tensor]] = []
        outside_label_id = int(label_to_id["O"])
        for example in examples:
            text = str(example.get("text") or "")
            encoding = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_offsets_mapping=True,
            )
            offsets = encoding.pop("offset_mapping")
            char_labels = _build_char_labels(example, text, label_to_id=label_to_id)
            token_labels = _labels_from_offsets(offsets, char_labels, outside_label_id=outside_label_id)
            self.items.append(
                {
                    "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
                    "labels": torch.tensor(token_labels, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.items[index]


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in batch[0]}


def _round_metric(value: float) -> float:
    return round(float(value), 4)


def evaluate_model(model, data_loader: DataLoader, device: torch.device) -> TrainingMetrics:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    true_positive = 0
    predicted_positive = 0
    actual_positive = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += float(outputs.loss.item())

            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            mask = labels != -100

            masked_predictions = predictions[mask]
            masked_labels = labels[mask]
            total_tokens += int(mask.sum().item())
            correct_tokens += int((masked_predictions == masked_labels).sum().item())

            pred_positive_mask = masked_predictions > 0
            true_positive_mask = masked_labels > 0
            true_positive += int(((masked_predictions == masked_labels) & true_positive_mask).sum().item())
            predicted_positive += int(pred_positive_mask.sum().item())
            actual_positive += int(true_positive_mask.sum().item())

    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = correct_tokens / total_tokens if total_tokens else 0.0
    average_loss = total_loss / max(1, len(data_loader))

    return TrainingMetrics(
        loss=_round_metric(average_loss),
        token_accuracy=_round_metric(accuracy),
        positive_precision=_round_metric(precision),
        positive_recall=_round_metric(recall),
        positive_f1=_round_metric(f1),
    )


def _save_label_mapping(output_dir: Path, label_to_id: dict[str, int], id_to_label: dict[int, str]) -> None:
    payload = {
        "label_to_id": label_to_id,
        "id_to_label": {str(key): value for key, value in id_to_label.items()},
    }
    with open(output_dir / "label_mapping.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def train_token_classifier(config: NERTrainingConfig) -> tuple[Path, TrainingSummary]:
    set_random_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(config.model_name_or_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Base model path not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for offset-based NER training")

    train_examples = _load_examples(config.train_path, max_examples=config.max_train_examples)
    dev_examples = _load_examples(config.dev_path, max_examples=config.max_dev_examples)
    if not train_examples:
        raise RuntimeError("NER training set is empty")
    if not dev_examples:
        raise RuntimeError("NER dev set is empty")

    label_to_id, id_to_label = build_label_mapping(train_examples + dev_examples)

    train_dataset = TokenClassificationDataset(train_examples, tokenizer, max_length=config.max_length, label_to_id=label_to_id)
    dev_dataset = TokenClassificationDataset(dev_examples, tokenizer, max_length=config.max_length, label_to_id=label_to_id)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=_collate)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=_collate)

    model = AutoModelForTokenClassification.from_pretrained(
        str(model_path),
        num_labels=len(label_to_id),
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
    )

    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_dev_f1 = -math.inf
    best_epoch = 0
    best_metrics: TrainingMetrics | None = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            running_loss += float(outputs.loss.item())

            should_step = step % config.gradient_accumulation_steps == 0 or step == len(train_loader)
            if should_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % config.log_every_steps == 0 or step == len(train_loader):
                logger.info(
                    "epoch=%s step=%s/%s train_loss=%.4f",
                    epoch,
                    step,
                    len(train_loader),
                    running_loss / step,
                )

        dev_metrics = evaluate_model(model, dev_loader, device=device)
        logger.info(
            "epoch=%s dev_loss=%.4f token_acc=%.4f positive_f1=%.4f",
            epoch,
            dev_metrics.loss,
            dev_metrics.token_accuracy,
            dev_metrics.positive_f1,
        )

        if dev_metrics.positive_f1 > best_dev_f1:
            best_dev_f1 = dev_metrics.positive_f1
            best_epoch = epoch
            best_metrics = dev_metrics
            model.save_pretrained(config.output_dir)
            tokenizer.save_pretrained(config.output_dir)
            _save_label_mapping(config.output_dir, label_to_id=label_to_id, id_to_label=id_to_label)

    if best_metrics is None:
        raise RuntimeError("Training finished without any dev metrics")

    summary = TrainingSummary(
        model_name_or_path=str(config.model_name_or_path),
        train_examples=len(train_examples),
        dev_examples=len(dev_examples),
        epochs=config.epochs,
        batch_size=config.batch_size,
        max_length=config.max_length,
        device=str(device),
        num_labels=len(label_to_id),
        label_names=[id_to_label[index] for index in sorted(id_to_label)],
        best_epoch=best_epoch,
        best_dev_positive_f1=_round_metric(best_metrics.positive_f1),
        best_dev_metrics=asdict(best_metrics),
    )
    with open(config.output_dir / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(summary), handle, ensure_ascii=False, indent=2)

    return config.output_dir, summary
