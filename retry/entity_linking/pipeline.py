from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .backends import (
    DictionaryMentionExtractor,
    HashingVectorizer,
    HybridMentionExtractor,
    NullVectorizer,
    TransformerMentionExtractor,
    TransformerVectorizer,
)
from .config import EntityLinkingConfig, resolve_first_existing
from .io_utils import iter_jsonl, read_description


logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    input_file: str
    output_file: str
    records_written: int = 0
    records_with_any_entity: int = 0
    records_with_zh_entities: int = 0
    records_with_en_entities: int = 0
    zh_vector_count: int = 0
    en_vector_count: int = 0


def build_extractor(config: EntityLinkingConfig, alias_records: Iterable[dict]):
    alias_records = list(alias_records)
    dictionary_extractor = DictionaryMentionExtractor.from_records(alias_records)

    if config.extractor == "dictionary":
        logger.info("Extractor backend: dictionary")
        return dictionary_extractor

    if config.extractor == "transformer":
        if not config.transformer_ner_model:
            raise RuntimeError("extractor=transformer requires --transformer-ner-model")
        logger.info("Extractor backend: transformer (%s)", config.transformer_ner_model)
        return HybridMentionExtractor(
            primary=TransformerMentionExtractor(config.transformer_ner_model),
            fallback=dictionary_extractor,
        )

    transformer_model = config.transformer_ner_model or resolve_first_existing(config.model_candidates.ner_models)
    if transformer_model is None:
        logger.warning("No local NER model found; falling back to dictionary extractor")
        return dictionary_extractor

    try:
        logger.info("Extractor backend: transformer (%s)", transformer_model)
        return HybridMentionExtractor(
            primary=TransformerMentionExtractor(transformer_model),
            fallback=dictionary_extractor,
        )
    except Exception as exc:  # pragma: no cover - dependent on local model state
        logger.warning("Failed to load transformer extractor (%s); falling back to dictionary extractor", exc)
        return dictionary_extractor


def build_vectorizer(config: EntityLinkingConfig):
    if config.vectorizer == "none":
        logger.info("Vector backend: none")
        return NullVectorizer()

    if config.vectorizer == "hash":
        logger.info("Vector backend: hash")
        return HashingVectorizer(dim=config.vector_dim)

    if config.vectorizer == "transformer":
        if not config.transformer_vector_model:
            raise RuntimeError("vectorizer=transformer requires --transformer-vector-model")
        logger.info("Vector backend: transformer (%s)", config.transformer_vector_model)
        return TransformerVectorizer(
            config.transformer_vector_model,
            dim=config.vector_dim,
            batch_size=config.vector_batch_size,
        )

    transformer_model = config.transformer_vector_model or resolve_first_existing(config.model_candidates.vector_models)
    if transformer_model is None:
        logger.warning("No local vector model found; falling back to no-vector mode")
        return NullVectorizer()

    try:
        logger.info("Vector backend: transformer (%s)", transformer_model)
        return TransformerVectorizer(
            transformer_model,
            dim=config.vector_dim,
            batch_size=config.vector_batch_size,
        )
    except Exception as exc:  # pragma: no cover - dependent on local model state
        logger.warning("Failed to load transformer vectorizer (%s); falling back to no-vector mode", exc)
        return NullVectorizer()


def process_record(record: dict, extractor, vectorizer, config: EntityLinkingConfig) -> dict:
    item = dict(record)

    zh_description = read_description(item, "zh")
    en_description = read_description(item, "en")

    zh_terms = []
    en_terms = []

    if len(zh_description.strip()) >= config.min_text_length:
        zh_terms = extractor.extract(zh_description, config.min_entity_length_zh)
    if len(en_description.strip()) >= config.min_text_length:
        en_terms = extractor.extract(en_description, config.min_entity_length_en)

    item["_entity_words_zh"] = zh_terms
    item["_entity_freq_zh"] = {term: 1 for term in zh_terms}
    item["_entity_count_zh"] = len(zh_terms)
    item["_entity_words_en"] = en_terms
    item["_entity_freq_en"] = {term: 1 for term in en_terms}
    item["_entity_count_en"] = len(en_terms)
    item["_entity_count_total"] = len(zh_terms) + len(en_terms)
    item["_entity_words_zh_vector"] = vectorizer.vectorize_terms(zh_terms)
    item["_entity_words_en_vector"] = vectorizer.vectorize_terms(en_terms)
    return item


def process_file(
    input_path: Path,
    output_path: Path,
    stats_path: Path,
    extractor,
    vectorizer,
    config: EntityLinkingConfig,
) -> ProcessingStats:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if output_path.exists() and not config.overwrite:
        raise FileExistsError(f"Output file already exists: {output_path} (use --overwrite)")

    stats = ProcessingStats(input_file=str(input_path), output_file=str(output_path))

    with open(output_path, "w", encoding="utf-8") as handle:
        for record in iter_jsonl(input_path, limit=config.max_records):
            processed = process_record(record, extractor, vectorizer, config)
            handle.write(json.dumps(processed, ensure_ascii=False) + "\n")
            stats.records_written += 1

            if processed["_entity_count_total"] > 0:
                stats.records_with_any_entity += 1
            if processed["_entity_count_zh"] > 0:
                stats.records_with_zh_entities += 1
            if processed["_entity_count_en"] > 0:
                stats.records_with_en_entities += 1
            if processed["_entity_words_zh_vector"] is not None:
                stats.zh_vector_count += 1
            if processed["_entity_words_en_vector"] is not None:
                stats.en_vector_count += 1

    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(asdict(stats), handle, ensure_ascii=False, indent=2)

    logger.info("Wrote %s records to %s", stats.records_written, output_path)
    return stats


def collect_alias_records(config: EntityLinkingConfig) -> list[dict]:
    records: list[dict] = []
    for path in (config.zh_input, config.en_input):
        records.extend(iter_jsonl(path, limit=config.max_records))
    return records


def run_pipeline(config: EntityLinkingConfig) -> tuple[ProcessingStats, ProcessingStats]:
    config.ensure_output_dir()
    alias_records = collect_alias_records(config)
    extractor = build_extractor(config, alias_records)
    vectorizer = build_vectorizer(config)

    zh_stats = process_file(
        input_path=config.zh_input,
        output_path=config.zh_output,
        stats_path=config.zh_stats_output,
        extractor=extractor,
        vectorizer=vectorizer,
        config=config,
    )
    en_stats = process_file(
        input_path=config.en_input,
        output_path=config.en_output,
        stats_path=config.en_stats_output,
        extractor=extractor,
        vectorizer=vectorizer,
        config=config,
    )
    return zh_stats, en_stats
