#!/bin/bash
# Full pipeline to fix missing vectors and rebuild both ES indexes.
#
# Step 1: Encode entity words with BGE-M3 → writes vectors into entity_words_zh/en.jsonl
# Step 2: Rebuild data2_enriched_t075_zhonly  (graph predictions, threshold=0.75)
# Step 3: Rebuild data2_enriched_kb_t075_zhonly (KB predictions,    threshold=0.75)
#
# Usage: bash retry/run_rebuild_indexes.sh [--skip-encode] [--skip-t075] [--skip-kb]
set -e
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export KMP_DUPLICATE_LIB_OK=TRUE

PYTHON=/d/software/anaconda/python
RETRY_DIR=/d/work/毕设/知识图谱/Knowledge-Graph/retry

SKIP_ENCODE=0
SKIP_T075=0
SKIP_KB=0
for arg in "$@"; do
    [[ "$arg" == "--skip-encode" ]] && SKIP_ENCODE=1
    [[ "$arg" == "--skip-t075" ]]  && SKIP_T075=1
    [[ "$arg" == "--skip-kb" ]]    && SKIP_KB=1
done

# ---------- Step 1: encode vectors ----------
if [[ $SKIP_ENCODE -eq 0 ]]; then
    echo "[$(date -Iseconds)] === Step 1: Encoding entity vectors with BGE-M3 ==="
    echo "  This will take ~30-45 min on CPU. Checkpoint cache is saved so it can resume."
    "$PYTHON" "$RETRY_DIR/run_encode_entity_vectors.py" --batch-size 16
    echo "[$(date -Iseconds)] Encoding complete."
else
    echo "[$(date -Iseconds)] Skipping encode step (--skip-encode)."
fi

# ---------- Step 2: rebuild data2_enriched_t075_zhonly (graph predictions) ----------
if [[ $SKIP_T075 -eq 0 ]]; then
    echo "[$(date -Iseconds)] === Step 2: Rebuilding data2_enriched_t075_zhonly ==="
    "$PYTHON" "$RETRY_DIR/run_enrich_index.py" \
        --predictions "$RETRY_DIR/output/alignment_predictions/bge_m3_graph_predictions.json" \
        --index-name data2_enriched_t075_zhonly \
        --score-threshold 0.75 \
        --no-unmatched-en \
        --output-dir "$RETRY_DIR/output/entity_linking/enriched_t075"
    echo "[$(date -Iseconds)] data2_enriched_t075_zhonly rebuilt."
else
    echo "[$(date -Iseconds)] Skipping t075 rebuild (--skip-t075)."
fi

# ---------- Step 3: rebuild data2_enriched_kb_t075_zhonly (KB predictions) ----------
if [[ $SKIP_KB -eq 0 ]]; then
    echo "[$(date -Iseconds)] === Step 3: Rebuilding data2_enriched_kb_t075_zhonly ==="
    "$PYTHON" "$RETRY_DIR/run_enrich_index.py" \
        --predictions "$RETRY_DIR/output/alignment_predictions/bge_m3_kb_predictions.json" \
        --index-name data2_enriched_kb_t075_zhonly \
        --score-threshold 0.75 \
        --no-unmatched-en \
        --output-dir "$RETRY_DIR/output/entity_linking/enriched_kb_t075"
    echo "[$(date -Iseconds)] data2_enriched_kb_t075_zhonly rebuilt."
else
    echo "[$(date -Iseconds)] Skipping KB rebuild (--skip-kb)."
fi

echo "[$(date -Iseconds)] === All done. Run evaluations next. ==="
echo ""
echo "  Validate:"
echo "    curl http://localhost:9200/data2_enriched_t075_zhonly/_count"
echo "    curl http://localhost:9200/data2_enriched_kb_t075_zhonly/_count"
echo ""
echo "  Re-run t075 eval (all 5 modes):"
echo "    KG_ES_INDEX_NAME=data2_enriched_t075_zhonly bash retry/run_kb_t075_eval.sh"
echo ""
echo "  Re-run KB eval (all 5 modes):"
echo "    bash retry/run_kb_t075_eval.sh"
