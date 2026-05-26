#!/usr/bin/env python3
"""
Predict zh->en alignments for our own KB entities via BGE-M3 cosine similarity.
Fully standalone — does NOT import alignment or vendor modules to avoid segfault.
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
DEFAULT_MODEL_DIR = RETRY_DIR / "models" / "alignment_baselines" / "bge-m3"
DEFAULT_ZH_INPUT = RETRY_DIR / "output" / "entity_linking" / "entity_words_zh.jsonl"
DEFAULT_EN_INPUT = RETRY_DIR / "output" / "entity_linking" / "entity_words_en.jsonl"
DEFAULT_OUTPUT = RETRY_DIR / "output" / "alignment_predictions" / "bge_m3_kb_predictions.json"


def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def entity_text(doc: dict, lang: str) -> str:
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
        parts.append(desc[:200])
    return " | ".join(parts) if parts else label or "<empty>"


def encode_texts(texts: list[str], model_dir: Path, batch_size: int = 8,
                 cache_file: Path | None = None) -> np.ndarray:
    """Encode with checkpointing — safe to restart if process crashes."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    # Resume from checkpoint
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

    print(f"  Loading model from {model_dir} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir))
    model.eval()

    with torch.no_grad():
        for i in range(start, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=256, return_tensors="pt")
            out = model(**enc)
            hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_vecs.append(pooled.detach().clone().numpy().astype(np.float32))
            del out, hidden, mask, pooled, enc
            # Checkpoint every 100 batches
            if cache_file and len(all_vecs) % 100 == 0:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zh-input", type=Path, default=DEFAULT_ZH_INPUT)
    parser.add_argument("--en-input", type=Path, default=DEFAULT_EN_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    cache_dir = args.output.parent / "encode_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Loading zh entities ...", flush=True)
    zh_docs = list(iter_jsonl(args.zh_input))
    zh_links = [str(d.get("wikipediaLink") or d.get("link") or "") for d in zh_docs]
    zh_texts = [entity_text(d, "zh") for d in zh_docs]
    print(f"  {len(zh_docs)} zh entities", flush=True)

    print("Loading en entities ...", flush=True)
    en_docs = list(iter_jsonl(args.en_input))
    en_links = [str(d.get("wikipediaLink") or d.get("link") or "") for d in en_docs]
    en_texts = [entity_text(d, "en") for d in en_docs]
    print(f"  {len(en_docs)} en entities", flush=True)

    all_texts = zh_texts + en_texts
    print(f"Encoding all {len(all_texts)} entities (batch={args.batch_size}) ...", flush=True)
    cache_file = cache_dir / "all_vecs.npy"
    all_mat = encode_texts(all_texts, args.model_dir, args.batch_size, cache_file=cache_file)
    zh_mat = all_mat[:len(zh_texts)]
    en_mat = all_mat[len(zh_texts):]

    print("Computing similarities ...", flush=True)
    scores_mat = zh_mat @ en_mat.T  # (N_zh, N_en)

    predictions = []
    for i in range(len(zh_docs)):
        best = int(np.argmax(scores_mat[i]))
        predictions.append({
            "zh_link": zh_links[i],
            "en_link": en_links[best],
            "score": float(scores_mat[i, best]),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(predictions)} predictions -> {args.output}", flush=True)

    sc = np.array([p["score"] for p in predictions])
    for t in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        print(f"  >= {t:.2f}: {(sc >= t).sum()} ({100*(sc >= t).mean():.1f}%)")


if __name__ == "__main__":
    main()
