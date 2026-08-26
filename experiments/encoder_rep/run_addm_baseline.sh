#!/usr/bin/env bash
# Detached train->validate chain for the aDDM BayesFlow baseline arm.
# Budget and batch size are the script's defaults, matched to addm_bf_apply.py.
#
#   TAG         output tag; checkpoints and figures land under item0_<TAG>
#   EXTRA_ARGS  extra flags forwarded to BOTH train and validate, e.g.
#               EXTRA_ARGS="--time_power_law_alpha 0"
#
# Runs against whatever bayesflow is installed -- it does NOT install or restore
# a pin, unlike run_addm_baseline_main.sh. Check the version before launching.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CASE=${CASE:-intercept_only}
FIX=${FIX:-item0}
TAG=${TAG:-best_defaults}
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
LOG="logs/real_data/addm_baseline_${TAG}.log"

echo "=== aDDM BF-baseline START $(date -Is) ===" | tee -a "$LOG"
echo "    tag=$TAG  extra='${EXTRA_ARGS:-}'" | tee -a "$LOG"
.venv/bin/python -c "import bayesflow; print('    bayesflow', bayesflow.__version__)" 2>/dev/null | tail -1 | tee -a "$LOG"

.venv/bin/python -m experiments.encoder_rep.addm_bf_baseline \
    --mode train --case "$CASE" --first_fixation "$FIX" \
    --tag "$TAG" "${EXTRA[@]}" >> "$LOG" 2>&1
rc=$?
echo "=== train exit=$rc $(date -Is) ===" | tee -a "$LOG"
if [ $rc -ne 0 ]; then echo "BASELINE TRAIN FAILED $(date -Is)" | tee -a "$LOG"; exit $rc; fi

.venv/bin/python -m experiments.encoder_rep.addm_bf_baseline \
    --mode validate --case "$CASE" --first_fixation "$FIX" \
    --tag "$TAG" "${EXTRA[@]}" >> "$LOG" 2>&1
echo "=== validate exit=$? $(date -Is) ===" | tee -a "$LOG"
echo "BASELINE ALL DONE $(date -Is)" | tee -a "$LOG"
