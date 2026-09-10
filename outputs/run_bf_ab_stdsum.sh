#!/usr/bin/env bash
# run_bf_ab_stdsum.sh — does standardizing summary_variables repair the baselines?
#
# WHY THIS IS NOW THE LEADING HYPOTHESIS (it was wrongly dismissed on 2026-09-09).
# The predecessor project's BF pipeline is near-identical to this one -- same
# adapter, same SetTransformer kwargs, same bare FlowMatching() -- and its
# baselines are EXCELLENT: calibration error 0.011-0.028, NRMSE 0.038-0.126, on
# 8x LESS training (1000 epochs x batch 64). Ours: 0.34-0.46 and 0.14-0.53.
#
# The one generative difference on intercept_only (no regressors, so design and
# coefficient settings are irrelevant) is CONTAMINATION, added here to match the
# CogFormer tiers. contaminate() replaces ~pi of trials with RT ~ Uniform(0, 20)
# against normal DDM RTs of ~0.3-2s. Measured on 256x500 trials:
#
#     channel   range            sd
#     design    [1.00, 1.00]     0.000
#     rts       [-1.00, 20.00]   1.638     <- 4% of trials > 3s, 1% > 10s
#     choices   [-1.00, 1.00]    0.360
#
# and BayesFlow standardizes ONLY inference_variables, so that heavy-tailed
# channel reaches the SetTransformer raw. The predecessor never hit this because
# without contamination its RTs were bounded and well-scaled.
#
# This also explains the two negative results: slicing the target (-3%) and the
# flow-matching alpha (-11%) both left the SUMMARY input untouched. And the
# predecessor fed the full grid WITH dead dims and still reached 0.019 on
# regressed, which independently confirms dead dims were never the cause.
#
# Single variable: --standardize_summaries. ddm/intercept_only, 300 epochs, so it
# 100 epochs: enough to show DIRECTION. The alpha and slicing arms ran 300, so
# absolute numbers are not comparable across experiments -- but the two arms here
# share a budget, which is what a single-variable A/B needs. BayesFlow
# checkpoints every epoch, so this can be extended by rerunning at a higher
# EPOCHS without losing anything.
#
# Arm logs are bf_ab_{raw,std}.log, deliberately NOT bf_ab_stdsum.log -- that is
# the driver's own stdout, and an arm named 'stdsum' collided with it.
#
# Usage: setsid nohup bash outputs/run_bf_ab_stdsum.sh > outputs/logs/bf_ab_stdsum.log 2>&1 &
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

log "A/B start: $MODEL/$CASE, $EPOCHS epochs, single variable = standardize_summaries"
arm raw 0 0-23  --no-standardize_summaries &
a=$!
arm std 1 24-47 --standardize_summaries &
b=$!
wait $a $b
log "A/B complete"
