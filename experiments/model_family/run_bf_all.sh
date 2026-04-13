#!/usr/bin/env bash
# Train all BayesFlow model-family cases for DDM, CDM, and RDM sequentially.
#
# Usage:
#   bash experiments/model_family/run_bf_all.sh
#   bash experiments/model_family/run_bf_all.sh --epochs 2000 --batch_size 128
#
# Any flags after the script name are forwarded to the training script only.
# Run from the project root directory.

set -euo pipefail

MODELS=("ddm" "cdm" "rdm")
CASES=("intercept_only" "fixed" "regressed" "fixed_regressed" "interaction" "full")
TRAIN_SCRIPT="experiments/model_family/model_family_bf_train.py"
TRAIN_ARGS=("$@")

SKIPPED=()
FAILED=()
PASSED=()

echo "============================================================"
echo " BayesFlow Model Family — train all models × cases"
echo " Models : ${MODELS[*]}"
echo " Cases  : ${CASES[*]}"
echo " Extra args: ${TRAIN_ARGS[*]:-(defaults)}"
echo "============================================================"

for model in "${MODELS[@]}"; do
    for case in "${CASES[@]}"; do

        tag="${model}/${case}"

        echo ""
        echo "------------------------------------------------------------"
        echo " SIMULATOR CHECK  :  $tag"
        echo "------------------------------------------------------------"
        if ! python "$TRAIN_SCRIPT" --model "$model" --case "$case" --test; then
            echo "  [SKIP] Simulator check failed — $tag"
            SKIPPED+=("$tag")
            continue
        fi

        echo ""
        echo "------------------------------------------------------------"
        echo " TRAINING         :  $tag"
        echo "------------------------------------------------------------"
        if python "$TRAIN_SCRIPT" --model "$model" --case "$case" "${TRAIN_ARGS[@]}"; then
            PASSED+=("$tag")
        else
            echo "  [FAIL] Training failed — $tag"
            FAILED+=("$tag")
        fi

    done
done

echo ""
echo "============================================================"
echo " Summary"
echo "============================================================"
echo " Passed  (${#PASSED[@]}) : ${PASSED[*]:-(none)}"
echo " Skipped (${#SKIPPED[@]}): ${SKIPPED[*]:-(none)}"
echo " Failed  (${#FAILED[@]}) : ${FAILED[*]:-(none)}"
echo "============================================================"
