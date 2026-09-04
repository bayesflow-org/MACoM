#!/usr/bin/env bash
# run_bf_cycle.sh — train the BayesFlow model-family baselines for ONE model on ONE
# GPU, cases sequentially. Two of these run concurrently, one per GPU, with disjoint
# CPU sets so the numba simulator pools do not oversubscribe.
#
#   Cycle 1:  ddm -> GPU 0 (cpus 0-23)      cdm -> GPU 1 (cpus 24-47)
#   Cycle 2:  rdm -> GPU 0 (cpus 0-23)      lbm -> (see notes; not supported here)
#
# Usage: bash outputs/run_bf_cycle.sh <model> <gpu> <cpus>
set -uo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2
PY=.venv/bin/python

MODEL=$1
GPU=$2
CPUS=$3

# "bench5" — the first five cases of run_bf_all.sh. `full` is deliberately excluded.
CASES=(intercept_only fixed regressed fixed_regressed interaction)

EPOCHS=2000        # overrides MF_BF_EPOCHS=1000 in experiments/shared_config.sh
STEPS=100
BATCH=256

# recommend_sim_workers() reads os.cpu_count() (48), NOT sched_getaffinity, so under
# taskset it would default to 44 workers on a 24-CPU partition and the two cycles
# would fight for cores. Pass the count explicitly: 20 workers + 4 cores for the main
# process and GPU feeding, matching the r4c4 CogFormer convention.
SIM_WORKERS=20

log() { echo "[$(date '+%F %T')] [$MODEL] $*"; }

log "start: gpu=$GPU cpus=$CPUS cases=${CASES[*]} epochs=$EPOCHS batch=$BATCH workers=$SIM_WORKERS"

for case in "${CASES[@]}"; do
    out="outputs/logs/bf_${MODEL}_${case}_$(date +%Y%m%d_%H%M%S).log"
    log "training case=$case -> $out"

    # XLA_PYTHON_CLIENT_PREALLOCATE=false: jax otherwise grabs 75% of the 49GB card up
    # front. This is a shared box -- co-tenants OOM when we hold memory we are not
    # using, and this network needs only a couple of GB.
    taskset -c "$CPUS" env CUDA_VISIBLE_DEVICES="$GPU" \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "$PY" -m experiments.model_family.model_family_bf_train \
        --model "$MODEL" --case "$case" \
        --epochs "$EPOCHS" --steps_per_epoch "$STEPS" --batch_size "$BATCH" \
        --sim_workers "$SIM_WORKERS" \
        --use_wandb > "$out" 2>&1

    rc=$?
    if [ $rc -eq 0 ]; then
        log "done case=$case"
    else
        log "FAILED case=$case (exit $rc) -- see $out; continuing to next case"
    fi
done

log "all cases complete"
