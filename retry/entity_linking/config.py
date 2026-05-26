from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() or (candidate / "work_wyy").exists():
            return candidate
    return start.resolve().parents[2]


REPO_ROOT = find_repo_root(Path(__file__).resolve())


def choose_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_ZH_INPUT = choose_existing(
    REPO_ROOT / "work_wyy" / "data" / "zh_wiki_v2.jsonl",
    REPO_ROOT / "data" / "raw" / "zh_wiki_v2.jsonl",
)
DEFAULT_EN_INPUT = choose_existing(
    REPO_ROOT / "work_wyy" / "data" / "en_wiki_v3.jsonl",
    REPO_ROOT / "data" / "raw" / "en_wiki_v3.jsonl",
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "retry" / "output" / "entity_linking"


@dataclass(frozen=True)
class ModelCandidates:
    ner_models: tuple[str, ...] = (
        str(REPO_ROOT / "data" / "models" / "ner_finetuned"),
        str(REPO_ROOT / "work_wyy" / "model" / "ner_finetuned"),
    )
    vector_models: tuple[str, ...] = (
        str(REPO_ROOT / "data" / "models" / "ner_finetuned"),
        str(REPO_ROOT / "work_wyy" / "model" / "ner_finetuned"),
        str(REPO_ROOT / "data" / "models" / "chinese-roberta-wwm-ext-large"),
        str(REPO_ROOT / "work_wyy" / "model" / "chinese-roberta-wwm-ext-large"),
    )


@dataclass(frozen=True)
class EntityLinkingConfig:
    repo_root: Path = REPO_ROOT
    zh_input: Path = DEFAULT_ZH_INPUT
    en_input: Path = DEFAULT_EN_INPUT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    extractor: str = "auto"
    vectorizer: str = "auto"
    max_records: int | None = None
    overwrite: bool = False
    min_text_length: int = 2
    min_entity_length_zh: int = 2
    min_entity_length_en: int = 3
    vector_dim: int = 1024
    vector_batch_size: int = 32
    transformer_ner_model: str | None = None
    transformer_vector_model: str | None = None
    model_candidates: ModelCandidates = field(default_factory=ModelCandidates)

    @property
    def zh_output(self) -> Path:
        return self.output_dir / "entity_words_zh.jsonl"

    @property
    def en_output(self) -> Path:
        return self.output_dir / "entity_words_en.jsonl"

    @property
    def zh_stats_output(self) -> Path:
        return self.output_dir / "entity_words_zh.stats.json"

    @property
    def en_stats_output(self) -> Path:
        return self.output_dir / "entity_words_en.stats.json"

    def ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


def resolve_first_existing(candidates: Iterable[str | Path]) -> str | None:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return str(path)
    return None

