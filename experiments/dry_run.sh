#!/usr/bin/env bash
# dry_run.sh — end-to-end smoke test for the full CogFormer experiment suite.
#
# Sets DRY_EPOCHS and related env vars, then delegates to run_all.sh.
# All sub-scripts pick up the env vars automatically and pass --dry_run_epochs
# to their training invocations.
#
# A good starting value for DRY_EPOCHS can be estimated as:
#   DRY_EPOCHS ≈ (target_hours × 3600) / (N_training_runs × steps_per_epoch × secs_per_step)
# Run with DRY_EPOCHS=2 as a pilot to measure secs_per_step, then scale up.
#
# Usage:
#   bash experiments/dry_run.sh [DRY_EPOCHS]
#   bash experiments/dry_run.sh 10
#   bash experiments/dry_run.sh 2>&1 | tee outputs/logs/dry_run.log

set -euo pipefail
mkdir -p outputs/logs outputs/timing

export DRY_EPOCHS="${1:-50}"      # CF training (default: 1% of 5000)
export MF_BF_DRY_EPOCHS=10       # BF training (1% of 1000)
export LCA_FFI_DRY_EPOCHS=5      # LCA/FFI BF training (1% of 500)
export USE_WANDB=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== CogFormer dry run: DRY_EPOCHS=${DRY_EPOCHS} ==="
echo "    Full-run budget: 5000 epochs. Extrapolation factor: $(echo "scale=1; 5000 / $DRY_EPOCHS" | bc)×"
echo ""

bash "${SCRIPT_DIR}/run_all.sh" 2>&1 | tee outputs/logs/dry_run.log
