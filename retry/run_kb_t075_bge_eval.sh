#!/bin/bash
# Eval pipeline using BGE-M3 (mean pooling) for vector search.
# Index: data2_enriched_kb_t075_bge
# This gives a fair comparison: both query and index use BGE-M3 mean pooling.
set -e
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export KMP_DUPLICATE_LIB_OK=TRUE

# BGE-M3 + mean pooling
export KG_VECTOR_MODEL_PATH="D:/work/毕设/知识图谱/Knowledge-Graph/retry/models/alignment_baselines/bge-m3"
export KG_VECTOR_POOLING=mean

export KG_ES_INDEX_NAME=data2_enriched_kb_t075_bge
export KG_EVAL_STATE_FILE="D:/work/毕设/知识图谱/Knowledge-Graph/retry/output/work_wyy_enriched_kb_t075_bge_eval_state.json"
export KG_EVAL_MAX_WORKERS=6
export KG_EVAL_LLM_WORKERS=2
export KG_EVAL_LLM_RETRIES=3
export KG_EVAL_LLM_RETRY_DELAY_SECONDS=5
export KG_ZHIPU_API_KEY=1a2a485fe1fc4bd5aa0d965bf452c8c8.se8RZdT8cH8skEDo

PYTHON=/d/software/anaconda/python
SCRIPT=/d/work/毕设/知识图谱/Knowledge-Graph/work_wyy/search_vllm.py
WORKDIR=/d/work/毕设/知识图谱/Knowledge-Graph/work_wyy

for MODE in --vector-only --es-text-only --llm-only --vector-llm-always --vector-llm; do
    echo "[$(date -Iseconds)] Starting mode: $MODE"
    cd "$WORKDIR" && "$PYTHON" "$SCRIPT" $MODE
    echo "[$(date -Iseconds)] Done: $MODE"
done
echo "[$(date -Iseconds)] ALL 5 MODES COMPLETE (BGE-M3)"
