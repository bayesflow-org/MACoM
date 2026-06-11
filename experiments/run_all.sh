#!/usr/bin/env bash
# run_all.sh — sequential orchestration of the full CogFormer experiment suite.
#
# Steps (in order):
#   1. model_family  — CF train/validate (DDM, CDM, RDM) + BF + C2ST
#   2. model_class   — CF train/validate + LBA validate
#   3. real_data     — apply all checkpoints to real data
#   4. generalization — CF train/validate (DDM × 4 holdout types)
#   5. encoder_rep   — CF train + LCA/FFI BF apply
#
# Ablations are run separately via experiments/ablations/run_ablations.sh.
#
# Environment variables (forwarded to sub-scripts automatically):
#   DRY_EPOCHS         — dry-run epoch count for CF training
#   MF_BF_DRY_EPOCHS   — dry-run epoch count for BF training
#   LCA_FFI_DRY_EPOCHS — dry-run epoch count for LCA/FFI BF training
#   USE_WANDB          — set to any non-empty value to enable wandb logging
#
# Usage:
#   bash experiments/run_all.sh 2>&1 | tee outputs/logs/run_all.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p outputs/logs outputs/timing

# Detect total number of GPUs on this machine.
N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)

echo "=== CogFormer experiment suite (N_GPUS=${N_GPUS}) ==="
echo ""

if [ "$N_GPUS" -ge 2 ]; then
    # ModelFamily (step 1) and ModelClass (step 2) are independent; run them
    # on separate GPUs in parallel.  Each sub-script sees only its own GPU via
    # CUDA_VISIBLE_DEVICES, so internal GPU-parallel logic within model_family
    # is suppressed (it would only use 1 GPU per job anyway).
    echo "--- Steps 1+2: ModelFamily (GPU 0) || ModelClass (GPU 1) ---"
    CUDA_VISIBLE_DEVICES=0 bash "${SCRIPT_DIR}/model_family/run.sh" \
        2>&1 | sed 's/^/[model_family] /' &
    pid_mf=$!
    CUDA_VISIBLE_DEVICES=1 bash "${SCRIPT_DIR}/model_class/run.sh" \
        2>&1 | sed 's/^/[model_class]  /' &
    pid_mc=$!
    wait $pid_mf $pid_mc
else
    echo "--- Step 1: ModelFamily ---"
    bash "${SCRIPT_DIR}/model_family/run.sh"

    echo "--- Step 2: ModelClass ---"
    bash "${SCRIPT_DIR}/model_class/run.sh"
fi

echo "--- Step 3: Real data ---"
bash "${SCRIPT_DIR}/real_data/run.sh"

echo "--- Step 4: Generalization ---"
bash "${SCRIPT_DIR}/generalization/run.sh"

echo "--- Step 5: Encoder Rep ---"
bash "${SCRIPT_DIR}/encoder_rep/run.sh"

echo ""
echo "=== All experiments complete. ==="
python -m experiments.timing_report
