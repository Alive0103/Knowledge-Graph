from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from entity_linking.backends import DictionaryMentionExtractor
from entity_linking.config import DEFAULT_EN_INPUT, DEFAULT_OUTPUT_DIR, DEFAULT_ZH_INPUT, REPO_ROOT
from entity_linking.io_utils import iter_jsonl, read_description, write_jsonl


DEFAULT_TRAINING_DATA_DIR = DEFAULT_OUTPUT_DIR.parent / "entity_linking_training"
DEFAULT_SUPERVISED_TRAINDATA_DIR = REPO_ROOT / "work_wyy" / "data" / "traindata"


@dataclass(frozen=True)
class WeakNERDataConfig:
    zh_input: Path = DEFAULT_ZH_INPUT
    en_input: Path = DEFAULT_EN_INPUT
    output_dir: Path = DEFAULT_TRAINING_DATA_DIR
    max_records: int | None = None
    train_ratio: float = 0.9
    negative_sample_ratio: float = 0.25
    min_text_length: int = 2
    min_entity_length_zh: int = 2
    min_entity_length_en: int = 3
    random_seed: int = 42

    @property
    def train_output(self) -> Path:
        return self.output_dir / "weak_ner_train.jsonl"

    @property
    def dev_output(self) -> Path:
        return self.output_dir / "weak_ner_dev.jsonl"

    @property
    def stats_output(self) -> Path:
        return self.output_dir / "weak_ner_stats.json"

    def ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class WeakNERStats:
    alias_count: int
    total_examples: int
    positive_examples: int
    negative_examples: int
    zh_examples: int
    en_examples: int
    train_examples: int
    dev_examples: int
    max_records: int | None


@dataclass(frozen=True)
class SupervisedNERDataConfig:
    source_dir: Path = DEFAULT_SUPERVISED_TRAINDATA_DIR
    output_dir: Path = DEFAULT_TRAINING_DATA_DIR
    max_train_examples: int | None = None
    max_dev_examples: int | None = None
    max_test_examples: int | None = None
    random_seed: int = 42

    @property
    def train_output(self) -> Path:
        return self.output_dir / "supervised_ner_train.jsonl"

    @property
    def dev_output(self) -> Path:
        return self.output_dir / "supervised_ner_dev.jsonl"

    @property
    def test_output(self) -> Path:
        return self.output_dir / "supervised_ner_test.jsonl"

    @property
    def stats_output(self) -> Path:
        return self.output_dir / "supervised_ner_stats.json"

    def ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class SupervisedNERStats:
    source_dir: str
    train_files: int
    dev_files: int
    test_files: int
    train_examples: int
    dev_examples: int
    test_examples: int
    entity_type_count: int
    entity_types: list[str]


def has_supervised_traindata(source_dir: Path = DEFAULT_SUPERVISED_TRAINDATA_DIR) -> bool:
    path = Path(source_dir)
    return path.exists() and any(path.glob("*_ner_train.json")) and any(path.glob("*_ner_dev.json"))


def _collect_alias_records(config: WeakNERDataConfig) -> list[dict]:
    records: list[dict] = []
    for path in (config.zh_input, config.en_input):
        records.extend(iter_jsonl(path, limit=config.max_records))
    return records


def _build_example(
    record: dict,
    language: str,
    description: str,
    spans,
    index: int,
) -> dict:
    return {
        "id": f"{language}:{index}",
        "language": language,
        "source_label": str(record.get("label") or ""),
        "text": description,
        "entities": [
            {"start": span.start, "end": span.end, "text": span.text, "label": "ENTITY"}
            for span in spans
        ],
        "entity_count": len(spans),
    }


def build_weak_ner_dataset(config: WeakNERDataConfig) -> tuple[Path, Path, WeakNERStats]:
    config.ensure_output_dir()
    alias_records = _collect_alias_records(config)
    extractor = DictionaryMentionExtractor.from_records(alias_records)

    positives: list[dict] = []
    negatives: list[dict] = []
    zh_count = 0
    en_count = 0

    for language, input_path, min_entity_length in (
        ("zh", config.zh_input, config.min_entity_length_zh),
        ("en", config.en_input, config.min_entity_length_en),
    ):
        for index, record in enumerate(iter_jsonl(input_path, limit=config.max_records)):
            description = read_description(record, language).strip()
            if len(description) < config.min_text_length:
                continue

            spans = extractor.extract_spans(description, min_length=min_entity_length)
            example = _build_example(record, language, description, spans, index)
            if spans:
                positives.append(example)
                if language == "zh":
                    zh_count += 1
                else:
                    en_count += 1
            else:
                negatives.append(example)

    rng = random.Random(config.random_seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    negative_keep = min(len(negatives), int(len(positives) * config.negative_sample_ratio))
    selected = positives + negatives[:negative_keep]
    rng.shuffle(selected)

    if not selected:
        raise RuntimeError("Weak NER dataset generation produced zero examples")

    train_size = max(1, int(len(selected) * config.train_ratio))
    if train_size >= len(selected):
        train_size = max(1, len(selected) - 1)
    train_examples = selected[:train_size]
    dev_examples = selected[train_size:]
    if not dev_examples:
        dev_examples = train_examples[-1:]
        train_examples = train_examples[:-1] or train_examples

    write_jsonl(config.train_output, train_examples)
    write_jsonl(config.dev_output, dev_examples)

    stats = WeakNERStats(
        alias_count=len(extractor.entries),
        total_examples=len(selected),
        positive_examples=len(positives),
        negative_examples=negative_keep,
        zh_examples=zh_count,
        en_examples=en_count,
        train_examples=len(train_examples),
        dev_examples=len(dev_examples),
        max_records=config.max_records,
    )
    with open(config.stats_output, "w", encoding="utf-8") as handle:
        json.dump(asdict(stats), handle, ensure_ascii=False, indent=2)

    return config.train_output, config.dev_output, stats


def _normalize_entity_type(raw_value: object) -> str:
    candidate = str(raw_value or "").strip()
    return candidate or "ENTITY"


def _convert_supervised_record(item: dict, source_file: str, row_index: int) -> tuple[dict | None, set[str]]:
    text = str(item.get("text") or "")
    if not text:
        return None, set()

    labels = ["O"] * len(text)
    entity_types: set[str] = set()
    for entity in list(item.get("entities") or []):
        start = max(0, min(len(text), int(entity.get("start", -1))))
        end = max(start, min(len(text), int(entity.get("end", -1))))
        if end <= start:
            continue
        entity_type = _normalize_entity_type(entity.get("type") or entity.get("label"))
        entity_types.add(entity_type)
        labels[start] = f"B-{entity_type}"
        for index in range(start + 1, end):
            labels[index] = f"I-{entity_type}"

    sample_id = item.get("sample_id")
    record = {
        "id": f"{source_file}:{sample_id if sample_id is not None else row_index}",
        "source_file": source_file,
        "sample_id": sample_id,
        "text": text,
        "labels": labels,
    }
    return record, entity_types


def _collect_entity_types_from_examples(examples: list[dict]) -> set[str]:
    entity_types: set[str] = set()
    for example in examples:
        for raw_label in list(example.get("labels") or []):
            label = str(raw_label or "O").strip()
            if label.startswith(("B-", "I-")) and len(label) > 2:
                entity_types.add(label[2:])
    return entity_types


def _load_supervised_split(
    source_dir: Path,
    pattern: str,
    limit: int | None = None,
    random_seed: int = 42,
) -> tuple[list[dict], set[str], int]:
    examples: list[dict] = []
    files = sorted(Path(source_dir).glob(pattern))
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            continue
        for row_index, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                continue
            record, _ = _convert_supervised_record(item, file_path.name, row_index)
            if record is None:
                continue
            examples.append(record)
    if limit is not None and len(examples) > limit:
        rng = random.Random(f"{pattern}:{random_seed}")
        rng.shuffle(examples)
        examples = examples[:limit]
    entity_types = _collect_entity_types_from_examples(examples)
    return examples, entity_types, len(files)


def build_supervised_ner_dataset(config: SupervisedNERDataConfig) -> tuple[Path, Path, Path, SupervisedNERStats]:
    config.ensure_output_dir()
    source_dir = Path(config.source_dir)
    if not has_supervised_traindata(source_dir):
        raise FileNotFoundError(f"Supervised traindata not found or incomplete: {source_dir}")

    train_examples, train_types, train_files = _load_supervised_split(
        source_dir=source_dir,
        pattern="*_ner_train.json",
        limit=config.max_train_examples,
        random_seed=config.random_seed,
    )
    dev_examples, dev_types, dev_files = _load_supervised_split(
        source_dir=source_dir,
        pattern="*_ner_dev.json",
        limit=config.max_dev_examples,
        random_seed=config.random_seed,
    )
    test_examples, test_types, test_files = _load_supervised_split(
        source_dir=source_dir,
        pattern="*_ner_test.json",
        limit=config.max_test_examples,
        random_seed=config.random_seed,
    )

    if not train_examples:
        raise RuntimeError("Supervised NER train split is empty")
    if not dev_examples:
        raise RuntimeError("Supervised NER dev split is empty")

    write_jsonl(config.train_output, train_examples)
    write_jsonl(config.dev_output, dev_examples)
    write_jsonl(config.test_output, test_examples)

    entity_types = sorted(train_types | dev_types | test_types)
    stats = SupervisedNERStats(
        source_dir=str(source_dir),
        train_files=train_files,
        dev_files=dev_files,
        test_files=test_files,
        train_examples=len(train_examples),
        dev_examples=len(dev_examples),
        test_examples=len(test_examples),
        entity_type_count=len(entity_types),
        entity_types=entity_types,
    )
    with open(config.stats_output, "w", encoding="utf-8") as handle:
        json.dump(asdict(stats), handle, ensure_ascii=False, indent=2)

    return config.train_output, config.dev_output, config.test_output, stats
