#!/usr/bin/env python3
"""
Generate top-1 alignment predictions for all Chinese entities using the trained
BGE-M3 + neighbor graph model, and save them as a JSON list.

Output format (list of dicts):
  [{"zh_id": 123, "zh_link": "https://zh.wikipedia.org/wiki/...",
    "en_id": 456, "en_link": "https://en.wikipedia.org/wiki/...",
    "score": 0.87}, ...]

Usage:
  python retry/run_predict_alignments.py [--device cpu|cuda] [--batch-size 128]
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
if str(RETRY_DIR) not in sys.path:
    sys.path.insert(0, str(RETRY_DIR))

from alignment.dbp15k import DBP15KDataset
from alignment.evaluation import encode_alignment_model
from alignment.model import ModelArgs, create_alignment_model, load_checkpoint

DATASET_DIR = RETRY_DIR / "recovered" / "alignment" / "DBP15K" / "zh_en"
MODEL_PATH = (
    RETRY_DIR
    / "output"
    / "alignment_training"
    / "bge_m3_neighbor_retrained_zh_en_overnight_complete_20260331_001_bge_graph"
    / "best_model.pth"
)
OUTPUT_DIR = RETRY_DIR / "output" / "alignment_predictions"
OUTPUT_FILE = OUTPUT_DIR / "bge_m3_graph_predictions.json"


def predict(device: str = "cpu", batch_size: int = 128, neighbor_size: int = 20) -> list[dict]:
    print(f"Loading dataset from {DATASET_DIR}")
    dataset = DBP15KDataset(dataset_dir=DATASET_DIR)

    print(f"Loading model from {MODEL_PATH}")
    model = create_alignment_model(device=device, args=ModelArgs(embedding_dim=1024))
    load_checkpoint(model, model_path=MODEL_PATH, device=device)

    print(f"Encoding KG1 (Chinese, {len(dataset.entities['1'])} entities) ...")
    left_ids, left_vecs = encode_alignment_model(
        dataset=dataset,
        kg="1",
        model=model,
        batch_size=batch_size,
        device=device,
        neighbor_size=neighbor_size,
        embedding_name="bge_m3",
    )
    print(f"  done: {len(left_ids)} vectors of dim {left_vecs.shape[1]}")

    print(f"Encoding KG2 (English, {len(dataset.entities['2'])} entities) ...")
    right_ids, right_vecs = encode_alignment_model(
        dataset=dataset,
        kg="2",
        model=model,
        batch_size=batch_size,
        device=device,
        neighbor_size=neighbor_size,
        embedding_name="bge_m3",
    )
    print(f"  done: {len(right_ids)} vectors of dim {right_vecs.shape[1]}")

    # Build entity-id → wikipedia-link maps
    zh_id_to_link: dict[int, str] = {
        eid: rec.name for eid, rec in dataset.entities["1"].items()
    }
    en_id_to_link: dict[int, str] = {
        eid: rec.name for eid, rec in dataset.entities["2"].items()
    }

    print("Computing top-1 predictions for all Chinese entities ...")
    predictions: list[dict] = []
    for i, zh_id in enumerate(left_ids):
        zh_id = int(zh_id)
        scores = right_vecs @ left_vecs[i]          # cosine sim (vectors are L2-normed)
        top_idx = int(np.argmax(scores))
        en_id = int(right_ids[top_idx])
        score = float(scores[top_idx])
        predictions.append(
            {
                "zh_id": zh_id,
                "zh_link": zh_id_to_link.get(zh_id, ""),
                "en_id": en_id,
                "en_link": en_id_to_link.get(en_id, ""),
                "score": round(score, 6),
            }
        )
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(left_ids)}")

    print(f"Generated {len(predictions)} predictions.")
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict entity alignments using BGE-M3+Graph model")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Torch device")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--neighbor-size", type=int, default=20)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()

    predictions = predict(
        device=args.device,
        batch_size=args.batch_size,
        neighbor_size=args.neighbor_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
