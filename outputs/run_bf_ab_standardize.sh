#!/usr/bin/env bash
# run_bf_ab_standardize.sh — controlled A/B: does standardizing summary_variables
# fix the BayesFlow baselines' overconfidence?
#
# The 2026-09 DDM/CDM baselines are severely miscalibrated: calibration error
# 0.34-0.46, posterior contraction 0.94-0.999, true value outside the posterior in
# ~90% of datasets, while CogFormer reaches 0.02-0.04 on the SAME simulations.
# Ruled out: axis swap, missing parameter standardization (BayesFlow does that by
# default), ODE under-integration (adaptive tsit5), degenerate masked dimensions.
# Remaining suspect: summary_variables never being standardized, so the
# SetTransformer sees raw RTs, binary choices and design columns on one scale.
#
# One variable changes between the arms. Both use RDM/intercept_only -- RDM is
# being retrained anyway, so nothing production depends on it -- and both write to
# TAGGED checkpoint directories so no existing baseline is touched.
#
# 300 epochs, not 2000: this asks whether calibration MOVES, not what the final
# number is. Both arms are equally undertrained, so the comparison is still fair.
#
# Usage: setsid nohup bash outputs/run_bf_ab_standardize.sh > outputs/logs/bf_ab.log 2>&1 &
set -uo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2
PY=.venv/bin/python

EPOCHS=300
MODEL=rdm
CASE=intercept_only

log() { echo "[$(date '+%F %T')] $*"; }

arm() {
    local name=$1 gpu=$2 cpus=$3 flag=$4
    local out="outputs/logs/bf_ab_${name}.log"
    log "[$name] gpu=$gpu $flag -> $out"
    taskset -c "$cpus" env CUDA_VISIBLE_DEVICES="$gpu" \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        NUMBA_THREADING_LAYER=workqueue MPLBACKEND=Agg \
        "$PY" -m experiments.model_family.model_family_bf_train \
        --model "$MODEL" --case "$CASE" \
        --epochs "$EPOCHS" --steps_per_epoch 100 --batch_size 256 \
        --sim_workers 20 "$flag" --tag "ab_${name}" > "$out" 2>&1
    log "[$name] exit=$?"
}

log "A/B start: $MODEL/$CASE, $EPOCHS epochs, one variable (summary standardization)"
arm control 0 0-23  --no-standardize_summaries &
a=$!
arm stdsum  1 24-47 --standardize_summaries &
b=$!
wait $a $b
log "A/B training complete -- run the calibration comparison next"
