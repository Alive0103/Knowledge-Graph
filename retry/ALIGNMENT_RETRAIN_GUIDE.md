# Alignment Retrain Guide

`retry/run_alignment_training.py` is the clean entry for retraining the `LaBSE + neighbor graph model` without touching the existing `data/models/final_model.pth`.

It now also supports `BGE-M3 + neighbor graph model` via `--embedding-name bge_m3`.

## New model naming

New retrained checkpoints are written under:

- `retry/output/alignment_training/labse_neighbor_retrained_<dataset>_<run_tag>/best_model.pth`
- `retry/output/alignment_training/labse_neighbor_retrained_<dataset>_<run_tag>/last_model.pth`
- `retry/output/alignment_training/bge_m3_neighbor_retrained_<dataset>_<run_tag>/best_model.pth`
- `retry/output/alignment_training/bge_m3_neighbor_retrained_<dataset>_<run_tag>/last_model.pth`

This keeps the old `final_model.pth` intact as a fallback baseline.

## Direct retraining

```bash
python retry/run_alignment_training.py \
  --dataset zh_en \
  --device cpu \
  --epochs 150
```

For `BGE-M3 + graph model`:

```bash
python retry/run_alignment_training.py \
  --dataset zh_en \
  --embedding-name bge_m3 \
  --device cpu \
  --epochs 150
```

## Evaluate the newly trained model

```bash
python retry/run_alignment.py eval \
  --mode final_model \
  --dataset zh_en \
  --device cpu \
  --model-path retry/output/alignment_training/labse_neighbor_retrained_zh_en_<run_tag>/best_model.pth \
  --json
```

For `BGE-M3 + graph model`, add `--embedding-name bge_m3` and point to the `bge_m3_neighbor_retrained_*` checkpoint.

## Full-chain automation

`retry/run_rigorous_full_experiment.py` supports three alignment-model strategies:

- `reuse`: only use an existing `final_model.pth`
- `train_if_missing`: retrain only when no usable final model exists
- `retrain`: always train a new `labse_neighbor_retrained_*` model and evaluate that new checkpoint

Example:

```bash
python retry/run_rigorous_full_experiment.py \
  --dataset zh_en \
  --device cpu \
  --alignment-model-strategy retrain \
  --alignment-epochs 150 \
  --include-bge-m3
```

The strict full pipeline will then run:

1. entity-linking NER training
   It now prefers supervised `work_wyy/data/traindata` when available and falls back to weak supervision only if that directory is missing.
2. entity-linking rebuild
3. ES indexing and evaluation
4. raw LaBSE baseline
5. retrained neighbor-graph alignment model
6. optional raw BGE-M3 baseline
7. teacher-facing comparison report
