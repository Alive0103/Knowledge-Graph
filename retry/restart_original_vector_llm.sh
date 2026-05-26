#!/bin/bash
# Restart --vector-llm for data2_original_bge
# Resumes from checkpoint ~116/444 with conservative rate limit
set -e
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export KMP_DUPLICATE_LIB_OK=TRUE

export KG_VECTOR_MODEL_PATH="D:/work/毕设/知识图谱/Knowledge-Graph/retry/models/alignment_baselines/bge-m3"
export KG_VECTOR_POOLING=mean
export KG_ES_INDEX_NAME=data2_original_bge
export KG_EVAL_STATE_FILE="D:/work/毕设/知识图谱/Knowledge-Graph/retry/output/work_wyy_original_bge_eval_state.json"
export KG_EVAL_MAX_WORKERS=6
export KG_EVAL_LLM_WORKERS=1
export KG_EVAL_LLM_RETRIES=3
export KG_EVAL_LLM_RETRY_DELAY_SECONDS=5
export KG_LLM_MIN_INTERVAL=5.0
export KG_ZHIPU_API_KEY=1a2a485fe1fc4bd5aa0d965bf452c8c8.se8RZdT8cH8skEDo

PYTHON=/d/software/anaconda/python
SCRIPT=/d/work/毕设/知识图谱/Knowledge-Graph/work_wyy/search_vllm.py
WORKDIR=/d/work/毕设/知识图谱/Knowledge-Graph/work_wyy
LOG="D:/work/毕设/知识图谱/Knowledge-Graph/retry/output/original_bge_eval.log"

echo "[$(date -Iseconds)] Restarting --vector-llm (1 worker, 5s interval, resume from ~116/444)" >> "$LOG"
cd "$WORKDIR" && "$PYTHON" "$SCRIPT" --vector-llm
echo "[$(date -Iseconds)] Done: --vector-llm" >> "$LOG"
echo "[$(date -Iseconds)] ALL 5 MODES COMPLETE (original BGE baseline)" >> "$LOG"
