#!/usr/bin/env bash
# run_bf_ab_slice.sh — does slicing the inference target to the ACTIVE cells fix
# the BayesFlow baselines' overconfidence?
#
# Case is ddm/regressed: 13 active cells out of a 28-cell grid, so 15 of the flow's
# 28 output dimensions are exactly 0.0 in every dataset. That is the largest
# shape-waste among the bench5 cases and therefore the sharpest test.
#
# Both arms use --time_power_law_alpha 0.0, so SLICING IS THE ONLY DIFFERENCE.
# (alpha was measured separately on rdm/intercept_only: 0.298 -> 0.265 calibration
# error, real but minor, so it is held fixed at the better value here rather than
# left as a second variable.)
#
# 300 epochs matches the alpha A/B, so all four arms are mutually comparable.
#
# Usage: setsid nohup bash outputs/run_bf_ab_slice.sh > outputs/logs/bf_ab_slice.log 2>&1 &
set -uo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2
PY=.venv/bin/python
EPOCHS=300; MODEL=ddm; CASE=regressed
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
        --sim_workers 20 --no-standardize_summaries --time_power_law_alpha 0.0 \
        "$@" --tag "ab_${name}" > "$out" 2>&1
    log "[$name] exit=$?"
}

log "A/B start: $MODEL/$CASE, $EPOCHS epochs, single variable = slice_active_target"
arm fullgrid 0 0-23  --no-slice_active_target &
a=$!
arm sliced   1 24-47 --slice_active_target &
b=$!
wait $a $b
log "A/B complete"
