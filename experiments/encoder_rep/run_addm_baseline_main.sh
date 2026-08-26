#!/usr/bin/env bash
# One-off control run: aDDM BF baseline under bayesflow MAIN (c4c0a59, v2.0.13)
# defaults, at the same budget as the best-defaults run, for a like-for-like
# comparison of the network defaults alone. Restores the project's pinned
# best-defaults install when finished, whatever the outcome.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=logs/real_data/addm_baseline_main_do0.log
PIN="bayesflow @ git+https://github.com/bayesflow-org/bayesflow.git@6842dcee3c0afaa319a0b578617bf5740061bc35"

restore() {
    echo "=== restoring best-defaults pin $(date -Is) ===" | tee -a "$LOG"
    VIRTUAL_ENV="$PWD/.venv" uv pip install --no-deps "$PIN" >> "$LOG" 2>&1
    .venv/bin/python -c "import bayesflow,sys; print('restored bayesflow', bayesflow.__version__)" 2>/dev/null | tail -1 | tee -a "$LOG"
}
trap restore EXIT

MAIN="bayesflow @ git+https://github.com/bayesflow-org/bayesflow.git@c4c0a594d67f5e32d9b5bdab008a0900d0f9210d"
echo "=== installing bayesflow MAIN $(date -Is) ===" | tee -a "$LOG"
VIRTUAL_ENV="$PWD/.venv" uv pip install --no-deps "$MAIN" >> "$LOG" 2>&1 || { echo "install failed" | tee -a "$LOG"; exit 1; }

echo "=== aDDM BF-baseline (MAIN defaults, TimeMLP dropout forced to 0) train START $(date -Is) ===" | tee -a "$LOG"
.venv/bin/python -m experiments.encoder_rep.addm_bf_baseline \
    --mode train --case intercept_only --first_fixation item0 \
    --tag main_do0 --subnet_dropout 0 >> "$LOG" 2>&1
rc=$?
echo "=== train exit=$rc $(date -Is) ===" | tee -a "$LOG"
[ $rc -ne 0 ] && { echo "MAIN-DEFAULTS TRAIN FAILED" | tee -a "$LOG"; exit $rc; }

.venv/bin/python -m experiments.encoder_rep.addm_bf_baseline \
    --mode validate --case intercept_only --first_fixation item0 \
    --tag main_do0 --subnet_dropout 0 >> "$LOG" 2>&1
echo "=== validate exit=$? $(date -Is) ===" | tee -a "$LOG"
echo "MAIN-DEFAULTS ALL DONE $(date -Is)" | tee -a "$LOG"
