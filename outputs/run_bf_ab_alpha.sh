#!/usr/bin/env bash
# run_bf_ab_alpha.sh — does the flow-matching time exponent explain the BF baselines'
# overconfidence?
#
# FINDING THIS TESTS. CogFormer and the BayesFlow baselines were optimising
# DIFFERENT flow-matching objectives:
#
#   CogFormer  cogformer/networks/transformers/decoder.py:230
#              t = torch.rand(...)                    -> t ~ U(0,1),  alpha = 0
#   BayesFlow  model_family_bf_train.py, bare FlowMatching()
#              time_power_law_alpha = 0.5 (library default in 2.0.13)
#              -> t = u**1.5, E[t] = 0.4
#
# Both interpolate theta_t = t*theta + (1-t)*z with target velocity theta - z, so
# t=1 is the DATA end -- exactly where the field learns to contract onto the
# posterior. alpha=0.5 biases training toward the noise end and starves that
# region. experiments/encoder_rep/addm_bf_baseline.py already names alpha=0.5 as
# "the leading suspect" for a calibration failure in this repo's other BF baseline.
#
# Both arms pass --no-standardize_summaries, matching the production config, so
# ALPHA IS THE ONLY DIFFERENCE. (Summary standardization was the earlier suspect,
# but CogFormer feeds raw RTs too -- cogformer/adapter/adapter.py normalizes only
# choices/alt/favored/theta -- so it cannot be what separates the two arms.)
#
# BayesFlow checkpoints every epoch, so the comparison can be run against partial
# checkpoints before the arms finish.
#
# Usage: setsid nohup bash outputs/run_bf_ab_alpha.sh > outputs/logs/bf_ab_alpha.log 2>&1 &
set -uo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2
PY=.venv/bin/python
EPOCHS=300; MODEL=rdm; CASE=intercept_only
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
        --sim_workers 20 --no-standardize_summaries \
        "$@" --tag "ab_${name}" > "$out" 2>&1
    log "[$name] exit=$?"
}

log "A/B start: $MODEL/$CASE, $EPOCHS epochs, single variable = time_power_law_alpha"
arm control 0 0-23  &                                  # library default alpha=0.5
a=$!
arm alpha0  1 24-47 --time_power_law_alpha 0.0 &       # matches CogFormer
b=$!
wait $a $b
log "A/B complete"
