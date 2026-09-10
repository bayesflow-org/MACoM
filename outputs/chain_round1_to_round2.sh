#!/usr/bin/env bash
# Score round 1 the moment it finishes, then start round 2 on the freed GPUs.
set -uo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
log() { echo "[$(date '+%F %T')] $*"; }
while ps -eo args | grep -q '[r]un_bf_ab_stdsum\.sh'; do sleep 60; done
log "round 1 finished"
for p in $(ps -eo pid,args | grep '[m]odel_family_bf_train' | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done
sleep 15
log "=== ROUND 1: standardize_summaries, single variable ==="
taskset -c 0-7 env CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg \
  .venv/bin/python outputs/bf_ab_score.py ddm intercept_only ab_raw ab_std 2>&1 \
  | grep -vE "^(Summarizing|Sampling|INFO|WARNING)"
log "=== launching round 2 (rt_transform=log) ==="
setsid nohup bash outputs/run_bf_ab_round2.sh > outputs/logs/bf_ab_round2_driver.log 2>&1 &
log "round 2 pid $!"
