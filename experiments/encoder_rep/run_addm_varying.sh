#!/usr/bin/env bash
# TRIAL-VARYING ITEM VALUES prototype -- both arms, then the shared-validation
# C2ST / log-density comparison. Sequential: one GPU. ~4h15m total.
#
# WHY: the v-fixed variant removed the STRUCTURAL flat direction in
# (v, v_diff, omega) but omega stayed unidentified (contraction 0.013), because
# the residual signal reaches the data only second-order, through the
# first-fixation transient's effect on RT shape, and free a/tau/s_v/s_tau absorb
# it. Here item values instead VARY BY TRIAL and are observed as regressors:
#     v_diff_t = b_d0 + b_d1*u_1t     v_t = b_v0 + b_v1*u_2t
# v_diff now enters BOTH the mu channel ((1+w)/2 * v_diff) and the Delta channel
# ((1-w)(2v + v_diff)) with the SAME b_d1, and the ratio of the two effects is
# 2(1-w)/(1+w) -- free of b_d1. So omega is structurally identified WITHOUT
# pinning v, and v is inferred again: the target is 9 cells, not 6.
#
# Honest caveat: structural identification is not practical identification. The
# Delta channel is still second-order, which is exactly what defeated the
# v-fixed variant. Cheap pre-checks could not settle this -- every one that pins
# the nuisance parameters overstates identifiability, because identifiability is
# a property of the likelihood AFTER a/tau/s_v/s_tau are free. Posterior
# contraction from a trained run is the measurement that settles it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source experiments/shared_config.sh

CASE=varying_values
EPOCHS_VF="${EPOCHS_VF:-500}"      # == the vfixed and 7-dim runs, so all are comparable
VAL_N="${VAL_N:-500}"
VAL_DRAWS="${VAL_DRAWS:-500}"
CHUNK="${CHUNK:-50}"
BASE_TAG="${BASE_TAG:-varying_e500}"
NJOBS="${NJOBS:-20}"
LOG="${LOG:-logs/real_data/addm_varying.log}"
mkdir -p "$(dirname "$LOG")"

FROZEN_CKPT="outputs/checkpoints/encoder_rep/bf/addm/${CASE}/addm_encoder_rep_fm/model.keras"
BASELINE_CKPT="outputs/checkpoints/encoder_rep/bf_baseline/addm/${CASE}/item0_${BASE_TAG}/addm_baseline_fm/model.keras"

skip_train() {
    if [ -z "${FORCE_TRAIN:-}" ] && [ -f "$1" ]; then
        echo "=== [$2] train SKIPPED (checkpoint present; FORCE_TRAIN=1 to override) $(date -Is) ===" | tee -a "$LOG"
        return 0
    fi
    return 1
}

.venv/bin/python -c "import bayesflow;print('bayesflow',bayesflow.__version__)" 2>/dev/null | tail -1 | tee -a "$LOG"
echo "=== VARYING-VALUES RUN epochs=$EPOCHS_VF val=${VAL_N}x${VAL_DRAWS} $(date -Is) ===" | tee -a "$LOG"

for mode in train validate; do
    [ "$mode" = train ] && skip_train "$FROZEN_CKPT" "frozen/varying" && continue
    echo "=== [frozen/varying] $mode START $(date -Is) ===" | tee -a "$LOG"
    .venv/bin/python -m experiments.encoder_rep.addm_bf_apply \
        --mode "$mode" --case "$CASE" --first_fixation item0 \
        --num_models 3 --encoder_device cuda --epochs "$EPOCHS_VF" \
        --val_batch_size "$VAL_N" --num_samples "$VAL_DRAWS" \
        --sample_chunk_size "$CHUNK" \
        --cogformer_checkpoint "$MC_CKPT" >>"$LOG" 2>&1
    rc=$?; echo "=== [frozen/varying] $mode exit=$rc $(date -Is) ===" | tee -a "$LOG"
    if [ $rc -ne 0 ]; then echo "FROZEN ARM FAILED" | tee -a "$LOG"; exit $rc; fi
done

echo "=== [frozen/varying] diagnostics START $(date -Is) ===" | tee -a "$LOG"
.venv/bin/python -m experiments.encoder_rep.addm_diagnostics --case "$CASE" \
    --data "./outputs/data/encoder_rep/addm/${CASE}/addm_encoder_rep_data.npz" >>"$LOG" 2>&1
echo "=== [frozen/varying] diagnostics exit=$? $(date -Is) ===" | tee -a "$LOG"

for mode in train validate; do
    [ "$mode" = train ] && skip_train "$BASELINE_CKPT" "baseline/varying" && continue
    echo "=== [baseline/varying] $mode START $(date -Is) ===" | tee -a "$LOG"
    .venv/bin/python -m experiments.encoder_rep.addm_bf_baseline \
        --mode "$mode" --case "$CASE" --first_fixation item0 \
        --tag "$BASE_TAG" --subnet_dropout 0 --epochs "$EPOCHS_VF" \
        --val_batch_size "$VAL_N" --num_samples "$VAL_DRAWS" \
        --sample_chunk_size "$CHUNK" >>"$LOG" 2>&1
    rc=$?; echo "=== [baseline/varying] $mode exit=$rc $(date -Is) ===" | tee -a "$LOG"
    if [ $rc -ne 0 ]; then echo "BASELINE ARM FAILED" | tee -a "$LOG"; exit $rc; fi
done

# Shared-validation comparison: paired log q(theta_true|x) and per-dataset C2ST
# against an empirical null. This is the step that showed the arms DIFFER on the
# v-fixed target despite matching on every marginal.
echo "=== [compare/varying] START $(date -Is) ===" | tee -a "$LOG"
.venv/bin/python -m experiments.encoder_rep.addm_arm_compare \
    --case "$CASE" --first_fixation item0 --tag "$BASE_TAG" --out_tag "$CASE" \
    --n_val "$VAL_N" --num_samples "$VAL_DRAWS" --sample_chunk_size "$CHUNK" \
    --n_c2st 200 --n_jobs "$NJOBS" \
    --cogformer_checkpoint "$MC_CKPT" >>"$LOG" 2>&1
rc=$?; echo "=== [compare/varying] exit=$rc $(date -Is) ===" | tee -a "$LOG"

echo "VARYING-VALUES ALL DONE $(date -Is)" | tee -a "$LOG"
