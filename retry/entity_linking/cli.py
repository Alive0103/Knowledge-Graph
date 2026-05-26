from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entity_linking.config import DEFAULT_EN_INPUT, DEFAULT_OUTPUT_DIR, DEFAULT_ZH_INPUT, EntityLinkingConfig
from entity_linking.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild entity-linking processed data from raw wiki JSONL files")
    parser.add_argument("--zh-input", type=Path, default=DEFAULT_ZH_INPUT, help="Path to zh_wiki_v2.jsonl")
    parser.add_argument("--en-input", type=Path, default=DEFAULT_EN_INPUT, help="Path to en_wiki_v3.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for processed JSONL output")
    parser.add_argument("--extractor", choices=("auto", "dictionary", "transformer"), default="auto")
    parser.add_argument("--vectorizer", choices=("auto", "none", "hash", "transformer"), default="auto")
    parser.add_argument("--transformer-ner-model", default=None, help="NER model path for extractor=transformer")
    parser.add_argument("--transformer-vector-model", default=None, help="Encoder model path for vectorizer=transformer")
    parser.add_argument("--max-records", type=int, default=None, help="Optional record limit for smoke tests")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--vector-dim", type=int, default=1024, help="Output vector dimension")
    parser.add_argument("--vector-batch-size", type=int, default=32, help="Batch size for transformer vectorization")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")

    config = EntityLinkingConfig(
        zh_input=args.zh_input,
        en_input=args.en_input,
        output_dir=args.output_dir,
        extractor=args.extractor,
        vectorizer=args.vectorizer,
        transformer_ner_model=args.transformer_ner_model,
        transformer_vector_model=args.transformer_vector_model,
        max_records=args.max_records,
        overwrite=args.overwrite,
        vector_dim=args.vector_dim,
        vector_batch_size=args.vector_batch_size,
    )

    zh_stats, en_stats = run_pipeline(config)

    print("Entity-linking rebuild completed")
    print(f"  zh output: {config.zh_output}")
    print(f"  en output: {config.en_output}")
    print(f"  zh records: {zh_stats.records_written}")
    print(f"  en records: {en_stats.records_written}")
    print(f"  zh vectors: {zh_stats.zh_vector_count}")
    print(f"  en vectors: {en_stats.en_vector_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

