#!/usr/bin/env bash
# run_bf_retrain.sh — retrain an EXPLICIT list of BF model-family cases for one model
# on one GPU, sequentially. Same knobs as run_bf_cycle.sh, but the case list is an
# argument instead of the hardcoded bench5, because only some cases are being redone.
#
# Why these cases (2026-09-04): two mask-randomizer bugs invalidated part of the grid.
#   - fixed / fixed_regressed: fixed_intrinsics (s_v, s_tau) were sampled instead of
#     pinned, because the category-count mask rebuild in ModelFamily.sample dropped
#     the fixed_params exclusion.
#   - interaction: pi never entered the parameter grid, because cases with an explicit
#     design_config bypass free_intrinsics, which is all _bf_cases_with_globals set.
# intercept_only and regressed simulated correctly throughout and are NOT retrained.
#
# Usage: bash outputs/run_bf_retrain.sh <model> <gpu> <cpus> <case> [case...]
set -uo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2
PY=.venv/bin/python

MODEL=$1; GPU=$2; CPUS=$3; shift 3
CASES=("$@")

EPOCHS=2000
STEPS=100
BATCH=256
SIM_WORKERS=20   # 20 workers + 4 cores for the main process, per 24-CPU partition

log() { echo "[$(date '+%F %T')] [$MODEL] $*"; }

log "retrain start: gpu=$GPU cpus=$CPUS cases=${CASES[*]} epochs=$EPOCHS batch=$BATCH"

for case in "${CASES[@]}"; do
    out="outputs/logs/bf_${MODEL}_${case}_$(date +%Y%m%d_%H%M%S).log"
    log "training case=$case -> $out"
    taskset -c "$CPUS" env CUDA_VISIBLE_DEVICES="$GPU" \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "$PY" -m experiments.model_family.model_family_bf_train \
        --model "$MODEL" --case "$case" \
        --epochs "$EPOCHS" --steps_per_epoch "$STEPS" --batch_size "$BATCH" \
        --sim_workers "$SIM_WORKERS" \
        --use_wandb > "$out" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then log "done case=$case"; else
        log "FAILED case=$case (exit $rc) -- see $out; continuing"
    fi
done

log "retrain complete: ${CASES[*]}"
