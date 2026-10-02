#!/bin/bash
# Stop `main` (pgid 2322326) once exp2 is done, so it never starts exp3 stages
# (exp3+exp4 already run separately on GPU 1). Then run e2_finalize ourselves.
cd /home/jerry/Documents/research/projects/CogFormer-2
source .venv/bin/activate
MAIN=2322326; LOG=logs/main_20260930_1013.out
echo "$(date '+%F %T') watching $LOG"
while kill -0 $MAIN 2>/dev/null; do
  if grep -q "── e2_finalize" "$LOG"; then
    echo "$(date '+%F %T') e2_validate done; stopping main"
    kill -TERM -- -$MAIN; sleep 10; kill -KILL -- -$MAIN 2>/dev/null
    MPLBACKEND=Agg bash run_experiments.sh e2_finalize && echo "$(date '+%F %T') e2_finalize ok"
    exit 0
  fi
  sleep 3
done
echo "$(date '+%F %T') main exited before e2_finalize (failure?) -- check $LOG"
