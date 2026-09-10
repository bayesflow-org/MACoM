#!/usr/bin/env bash
# chain_ab_to_full.sh — when the slicing A/B finishes: score it, then start the
# full BayesFlow retrain.
#
# Three steps, in order:
#   1. wait for outputs/run_bf_ab_slice.sh to exit
#   2. score both arms on one shared validation set -> the number that justifies
#      (or indicts) the fix this retrain is built on
#   3. launch outputs/run_bf_full.sh
#
# THE LAUNCH IS UNCONDITIONAL, by request. The score is printed first and lands in
# this log ~10 hours before the first case finishes, so there is ample room to
# abort if it reads badly:
#     pkill -f run_bf_full.sh    # then clear stragglers, see below
# Reference points for reading it: the production 2000-epoch ddm/regressed sits at
# 0.405 calibration error; CogFormer on the same simulations reaches 0.02-0.04.
#
# Scoring runs BEFORE the launch, not concurrently, because it loads two
# checkpoints onto GPU 0 -- the same card wave 1 immediately starts training on.
#
# Usage: setsid nohup bash outputs/chain_ab_to_full.sh > outputs/logs/chain_ab_to_full.log 2>&1 &
set -uo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PY=.venv/bin/python
log() { echo "[$(date '+%F %T')] $*"; }

if ps -eo args | grep -q '[r]un_bf_ab_slice\.sh'; then
    log "waiting for the slicing A/B to finish"
    while ps -eo args | grep -q '[r]un_bf_ab_slice\.sh'; do sleep 120; done
    log "A/B driver exited"
else
    log "A/B driver not running; treating it as already finished"
fi

# The driver is gone, but a trainer that outlived it would still hold a card and
# would then fight wave 1 for the same GPU.
for p in $(ps -eo pid,args | grep '[m]odel_family_bf_train --model ddm' | awk '{print $1}'); do
    log "killing straggler trainer pid $p"; kill -9 "$p" 2>/dev/null
done
sleep 20
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | while read -r l; do log "gpu $l"; done

log "=== scoring the slicing A/B (single variable: slice_active_target) ==="
taskset -c 0-7 env CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg \
    "$PY" outputs/bf_ab_slice_compare.py 2>&1 | grep -vE "^(Summarizing|Sampling|INFO|WARNING)" || \
    log "WARNING: scoring failed; launching anyway (results are still on disk)"

log "=== launching the full retrain ==="
# NOT `exec`: this script must stay alive as the parent so its log keeps the score
# and the launch in one place.
setsid nohup bash outputs/run_bf_full.sh > outputs/logs/bf_full.log 2>&1 &
log "run_bf_full.sh launched as pid $! -> outputs/logs/bf_full.log"
log "case order: intercept_only interaction fixed regressed fixed_regressed"
log "wave 1 ddm(GPU0)/cdm(GPU1), then wave 2 rdm(GPU0)/lba(GPU1); ~10h per case"
