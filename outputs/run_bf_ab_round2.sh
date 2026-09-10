#!/usr/bin/env bash
# run_bf_ab_round2.sh — second half of a 2x2 over the two summary-input fixes.
#
# GOAL: reach the predecessor's numbers. ../CogFormer's BF baselines hit
# calibration error 0.011-0.028 / NRMSE 0.038-0.126 on this same task with 8x LESS
# training. Ours sit at 0.34-0.46 / 0.14-0.53.
#
# The one generative difference on intercept_only (no regressors, so design and
# coefficient settings are moot) is CONTAMINATION -- pi of trials replaced with
# RT ~ Uniform(0,20). Measured over 256x500 trials:
#
#                 rts sd    p99     max     frac>10s
#   predecessor    0.736    3.94    10.4     0.0000
#   ours           1.666   10.15    20.0     0.0102
#
# and that channel reaches the SetTransformer raw, because BayesFlow standardizes
# only inference_variables. Two independent ways to fix the conditioning:
#
#   standardize_summaries : z-scores the channel (fixes SCALE; a 20s RT still
#                           lands at z ~ 11, so it does not fix the TAIL)
#   rt_transform=log      : log(rts + 2) (fixes the TAIL: 20s -> 3.09, channel
#                           sd 7.79 -> 1.03; stays finite on the rts=-1 sentinel)
#
# Round 1 ran raw+none and std+none. This is raw+log and std+log, completing:
#
#            rt_transform=none      rt_transform=log
#   raw          round 1                round 2
#   std          round 1                round 2
#
# All four at ddm/intercept_only, 100 epochs, alpha=0, sliced -- so only the two
# summary-conditioning variables move.
#
# Usage: setsid nohup bash outputs/run_bf_ab_round2.sh > outputs/logs/bf_ab_round2_driver.log 2>&1 &
set -uo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2
PY=.venv/bin/python
EPOCHS=100; MODEL=ddm; CASE=intercept_only
log() { echo "[$(date '+%F %T')] $*"; }

arm() {
    local name=$1 gpu=$2 cpus=$3; shift 3
    local out="outputs/logs/bf_ab_${name}.log"
    log "[$name] gpu=$gpu extra='$*' -> $out"
    taskset -c "$cpus" env CUDA_VISIBLE_DEVICES="$gpu" \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        NUMBA_THREADING_LAYER=workqueue MPLBACKEND=Agg \
        "$PY" -m experiments.model_family.model_family_bf_train \
        --model "$MODEL" --case "$CASE" \
        --epochs "$EPOCHS" --steps_per_epoch 100 --batch_size 256 \
        --sim_workers 20 --time_power_law_alpha 0.0 \
        "$@" --tag "ab_${name}" > "$out" 2>&1
    log "[$name] exit=$?"
}

log "round 2: $MODEL/$CASE, $EPOCHS epochs, rt_transform=log with and without summary standardization"
arm rawlog 0 0-23  --no-standardize_summaries --rt_transform log &
a=$!
arm stdlog 1 24-47 --standardize_summaries    --rt_transform log &
b=$!
wait $a $b
log "round 2 complete"
log "=== FULL 2x2 ==="
"$PY" outputs/bf_ab_score.py "$MODEL" "$CASE" ab_raw ab_std ab_rawlog ab_stdlog
