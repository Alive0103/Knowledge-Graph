#!/usr/bin/env python3
"""
Encode entity text with BGE-M3 and write vectors back into entity_words JSONL files.

Run BEFORE run_enrich_index.py to populate:
  _entity_words_zh_vector  (in entity_words_zh.jsonl)
  _entity_words_en_vector  (in entity_words_en.jsonl)

Uses mean pooling (same as run_predict_kb_alignments.py) and saves npy
checkpoints so the run can be resumed if interrupted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

RETRY_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = RETRY_DIR.parent / "work_wyy" / "model" / "ner_finetuned"
DEFAULT_ZH_INPUT = RETRY_DIR / "output" / "entity_linking" / "entity_words_zh.jsonl"
DEFAULT_EN_INPUT = RETRY_DIR / "output" / "entity_linking" / "entity_words_en.jsonl"
DEFAULT_CACHE_DIR = RETRY_DIR / "output" / "entity_linking" / "encode_cache"


def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def entity_text(doc: dict, lang: str) -> str:
    """Build a single text string from label + aliases + description."""
    parts = []
    label = str(doc.get("label") or "").strip()
    if label:
        parts.append(label)
    if lang == "zh":
        aliases = doc.get("zh_aliases") or doc.get("aliases_zh") or []
        desc = str(doc.get("zh_description") or doc.get("descriptions_zh") or "").strip()
    else:
        aliases = doc.get("en_aliases") or doc.get("aliases_en") or []
        desc = str(doc.get("en_description") or doc.get("descriptions_en") or "").strip()
    if isinstance(aliases, list):
        parts.extend(str(a).strip() for a in aliases[:3] if str(a).strip())
    if desc:
        parts.append(desc[:300])
    return " | ".join(parts) if parts else label or "<empty>"


def load_encoder(model_dir: Path):
    """Load encoder model. For NER models, extracts the base encoder (matching eval's vector_model.py)."""
    from transformers import AutoModel, AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    try:
        ner_model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
        encoder = getattr(ner_model, "bert", None) or getattr(ner_model, "base_model", None)
        if encoder is None:
            raise AttributeError("no encoder attribute")
        print(f"  Extracted encoder from NER model (hidden_size={encoder.config.hidden_size})", flush=True)
    except Exception:
        encoder = AutoModel.from_pretrained(str(model_dir))
        print(f"  Loaded base model (hidden_size={encoder.config.hidden_size})", flush=True)
    return tokenizer, encoder


def encode_texts(
    texts: list[str],
    model_dir: Path,
    batch_size: int = 32,
    cache_file: Path | None = None,
    pooling: str = "cls",
) -> np.ndarray:
    """Encode texts with CLS or mean pooling. Supports checkpoint resume.
    pooling='cls'  — matches eval's vector_model.py (NER model)
    pooling='mean' — mean pooling for retrieval models like BGE-M3
    """
    import torch

    if cache_file and cache_file.exists():
        cached = np.load(str(cache_file))
        start = cached.shape[0]
        if start >= len(texts):
            print(f"  all {start} vectors loaded from cache", flush=True)
            return cached
        print(f"  resuming from cache: {start}/{len(texts)}", flush=True)
        all_vecs = [cached]
    else:
        start = 0
        all_vecs = []

    print(f"  Loading model from {model_dir} (pooling={pooling}) ...", flush=True)
    tokenizer, model = load_encoder(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}", flush=True)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i in range(start, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            hidden = out.last_hidden_state
            if pooling == "mean":
                mask = enc["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            else:
                pooled = hidden[:, 0, :]
            all_vecs.append(pooled.detach().cpu().numpy().astype(np.float32))
            del out, hidden, pooled, enc

            if cache_file and (i // batch_size + 1) % 100 == 0:
                tmp = np.concatenate(all_vecs, axis=0)
                np.save(str(cache_file), tmp)

            if i % (batch_size * 25) == 0:
                done = min(i + batch_size, len(texts))
                print(f"  {done}/{len(texts)}", flush=True)

    mat = np.concatenate(all_vecs, axis=0)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    result = mat / np.where(norms == 0, 1.0, norms)
    if cache_file:
        np.save(str(cache_file), result)
    return result


def write_vectors_back(
    jsonl_path: Path,
    vector_key: str,
    vectors: np.ndarray,
    docs: list[dict],
) -> int:
    """Merge vectors into docs and overwrite the JSONL file. Returns non-null count."""
    count = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for doc, vec in zip(docs, vectors):
            if vec is not None and np.linalg.norm(vec) > 0:
                doc[vector_key] = vec.tolist()
                count += 1
            else:
                doc[vector_key] = None
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode entity vectors (supports NER/CLS and BGE-M3/mean)")
    parser.add_argument("--zh-input", type=Path, default=DEFAULT_ZH_INPUT)
    parser.add_argument("--en-input", type=Path, default=DEFAULT_EN_INPUT)
    parser.add_argument("--zh-output", type=Path, default=None,
                        help="Output path for ZH JSONL (default: overwrite input)")
    parser.add_argument("--en-output", type=Path, default=None,
                        help="Output path for EN JSONL (default: overwrite input)")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls",
                        help="cls=NER model (default), mean=retrieval models like BGE-M3")
    args = parser.parse_args()
    if args.zh_output is None:
        args.zh_output = args.zh_input
    if args.en_output is None:
        args.en_output = args.en_input

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------- zh entities: encode zh text → _entity_words_zh_vector ----------
    print(f"\n=== ZH entities: {args.zh_input} ===", flush=True)
    zh_docs = list(iter_jsonl(args.zh_input))
    print(f"  Loaded {len(zh_docs)} records", flush=True)
    zh_texts = [entity_text(d, "zh") for d in zh_docs]

    zh_cache = args.cache_dir / "zh_vectors.npy"
    zh_mat = encode_texts(zh_texts, args.model_dir, args.batch_size, cache_file=zh_cache, pooling=args.pooling)
    n = write_vectors_back(args.zh_output, "_entity_words_zh_vector", zh_mat, zh_docs)
    print(f"  Wrote {n}/{len(zh_docs)} zh vectors → {args.zh_output}", flush=True)

    # ---------- en entities: encode en text → _entity_words_en_vector ----------
    print(f"\n=== EN entities: {args.en_input} ===", flush=True)
    en_docs = list(iter_jsonl(args.en_input))
    print(f"  Loaded {len(en_docs)} records", flush=True)
    en_texts = [entity_text(d, "en") for d in en_docs]

    en_cache = args.cache_dir / "en_vectors.npy"
    en_mat = encode_texts(en_texts, args.model_dir, args.batch_size, cache_file=en_cache, pooling=args.pooling)
    n = write_vectors_back(args.en_output, "_entity_words_en_vector", en_mat, en_docs)
    print(f"  Wrote {n}/{len(en_docs)} en vectors → {args.en_output}", flush=True)

    print("\nDone. Next: python retry/run_enrich_index.py", flush=True)


if __name__ == "__main__":
    main()
