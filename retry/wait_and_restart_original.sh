#!/bin/bash
# Wait for BGE enriched eval to fully complete, then restart original BGE eval
LOG_ENRICHED="D:/work/毕设/知识图谱/Knowledge-Graph/retry/output/kb_t075_bge_eval.log"
LOG_ORIGINAL="D:/work/毕设/知识图谱/Knowledge-Graph/retry/output/original_bge_eval.log"

echo "[$(date -Iseconds)] Waiting for BGE enriched eval to complete..."
until grep -q "ALL 5 MODES COMPLETE" "$LOG_ENRICHED" 2>/dev/null; do
    sleep 60
done
echo "[$(date -Iseconds)] BGE enriched eval complete. Restarting original BGE eval..."
bash /d/work/毕设/知识图谱/Knowledge-Graph/retry/run_original_bge_eval.sh >> "$LOG_ORIGINAL" 2>&1
echo "[$(date -Iseconds)] Original BGE eval restarted and complete."
