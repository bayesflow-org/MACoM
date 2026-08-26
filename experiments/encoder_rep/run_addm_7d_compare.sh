#!/usr/bin/env bash
# best-defaults vs main, BayesFlow-only aDDM arm, 7-dim inference target.
# TimeMLP dropout is forced to 0 on BOTH arms: it is the one default the two
# libraries disagree on that is not architectural (branch 0.0, main 0.05), so
# equalising it keeps the comparison about the architecture. Everything else is
# left at each library's own default. Restores the project pin at the end.
EPOCHS=${EPOCHS:-200}
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=logs/real_data/addm_7d_compare.log
BRANCH="bayesflow @ git+https://github.com/bayesflow-org/bayesflow.git@6842dcee3c0afaa319a0b578617bf5740061bc35"
MAIN="bayesflow @ git+https://github.com/bayesflow-org/bayesflow.git@c4c0a594d67f5e32d9b5bdab008a0900d0f9210d"

restore() { echo "=== restoring project pin (best-defaults) $(date -Is) ===" | tee -a "$LOG"
            VIRTUAL_ENV="$PWD/.venv" uv pip install --no-deps "$BRANCH" >>"$LOG" 2>&1; }
trap restore EXIT

arm () {   # arm <pin> <tag>
    echo "=== [$2] installing bayesflow $(date -Is) ===" | tee -a "$LOG"
    VIRTUAL_ENV="$PWD/.venv" uv pip install --no-deps "$1" >>"$LOG" 2>&1 || return 1
    .venv/bin/python -c "import bayesflow;print('    bayesflow',bayesflow.__version__)" 2>/dev/null | tail -1 | tee -a "$LOG"
    for mode in train validate; do
        echo "=== [$2] $mode START $(date -Is)  (epochs=$EPOCHS, dropout=0) ===" | tee -a "$LOG"
        .venv/bin/python -m experiments.encoder_rep.addm_bf_baseline \
            --mode "$mode" --case intercept_only --first_fixation item0 \
            --tag "$2" --subnet_dropout 0 --epochs "$EPOCHS" >>"$LOG" 2>&1
        rc=$?; echo "=== [$2] $mode exit=$rc $(date -Is) ===" | tee -a "$LOG"
        if [ $rc -ne 0 ]; then return $rc; fi
    done
    # Explicit: without this the function inherits the status of the last
    # command in the loop body. `[ 0 -ne 0 ] && return` is FALSE on success, so
    # `arm` returned 1 after a clean run and the caller's || branch fired.
    return 0
}

arm "$BRANCH" bd_7d   || { echo "BRANCH ARM FAILED" | tee -a "$LOG"; exit 1; }
arm "$MAIN"   main_7d || { echo "MAIN ARM FAILED"   | tee -a "$LOG"; exit 1; }
echo "7D COMPARE ALL DONE $(date -Is)" | tee -a "$LOG"
