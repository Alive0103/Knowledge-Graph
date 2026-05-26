#!/bin/bash
set -e
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export KMP_DUPLICATE_LIB_OK=TRUE
export KG_ES_INDEX_NAME=data2_enriched_t075_zhonly
export KG_EVAL_STATE_FILE="D:/work/毕设/知识图谱/Knowledge-Graph/retry/output/work_wyy_enriched_t075_zhonly_eval_state.json"
export KG_EVAL_MAX_WORKERS=6
export KG_EVAL_LLM_WORKERS=2
export KG_EVAL_LLM_RETRIES=3
export KG_EVAL_LLM_RETRY_DELAY_SECONDS=5
export KG_ZHIPU_API_KEY=1a2a485fe1fc4bd5aa0d965bf452c8c8.se8RZdT8cH8skEDo

PYTHON=/d/software/anaconda/python
SCRIPT=/d/work/毕设/知识图谱/Knowledge-Graph/work_wyy/search_vllm.py
WORKDIR=/d/work/毕设/知识图谱/Knowledge-Graph/work_wyy

echo "[$(date -Iseconds)] Starting llm_only (resuming from checkpoint)"
cd "$WORKDIR" && "$PYTHON" "$SCRIPT" --llm-only
echo "[$(date -Iseconds)] llm_only done"

echo "[$(date -Iseconds)] Starting vector_with_llm_always"
cd "$WORKDIR" && "$PYTHON" "$SCRIPT" --vector-llm-always
echo "[$(date -Iseconds)] vector_with_llm_always done"

echo "[$(date -Iseconds)] Starting vector_with_llm"
cd "$WORKDIR" && "$PYTHON" "$SCRIPT" --vector-llm
echo "[$(date -Iseconds)] vector_with_llm done — ALL MODES COMPLETE"
