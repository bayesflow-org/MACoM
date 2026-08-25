#!/usr/bin/env bash
# Detached train->validate chain for the aDDM BayesFlow baseline arm.
# Budget and batch size are the script's defaults, matched to addm_bf_apply.py.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=logs/real_data/addm_baseline.log
CASE=${CASE:-intercept_only}
FIX=${FIX:-item0}

echo "=== aDDM BF-baseline train START $(date -Is) ===" | tee -a "$LOG"
.venv/bin/python -m experiments.encoder_rep.addm_bf_baseline \
    --mode train --case "$CASE" --first_fixation "$FIX" >> "$LOG" 2>&1
rc=$?
echo "=== train exit=$rc $(date -Is) ===" | tee -a "$LOG"
if [ $rc -ne 0 ]; then echo "BASELINE TRAIN FAILED $(date -Is)" | tee -a "$LOG"; exit $rc; fi

.venv/bin/python -m experiments.encoder_rep.addm_bf_baseline \
    --mode validate --case "$CASE" --first_fixation "$FIX" >> "$LOG" 2>&1
echo "=== validate exit=$? $(date -Is) ===" | tee -a "$LOG"
echo "BASELINE ALL DONE $(date -Is)" | tee -a "$LOG"
