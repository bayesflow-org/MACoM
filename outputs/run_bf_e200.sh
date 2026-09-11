#!/usr/bin/env bash
# 200-epoch BF probe for DDM + CDM under the 2026-09-10 changes:
#   * standardize_summaries now OFF by default (inference_variables only, as in the
#     predecessor project) -- passed explicitly here so the run is self-documenting
#   * bf_validate pins ModelFamily(num_workers=1), so seed_case actually pins the
#     validation set
#
# WHY 200 EPOCHS. Calibration error on these baselines degrades monotonically with
# training length while the posterior MEANS stay accurate and the spread collapses:
#
#     ddm/intercept_only    100 ep  calib 0.145  contraction 0.927
#     ddm/intercept_only   2000 ep  calib 0.388  contraction 0.992
#     rdm/intercept_only    300 ep  calib 0.295  contraction 0.953
#     rdm/intercept_only   2000 ep  calib 0.432  contraction 0.995
#
# 200 epochs sits in the region where posterior variance should still be honestly
# large, which is the thing being checked.
#
# EVERYTHING IS TAGGED __e200. Production checkpoints, figures, npz and evaluation
# CSVs are at the untagged paths and are NOT touched, so 200 ep and 2000 ep can be
# compared side by side afterwards.
#
# Usage: setsid nohup bash outputs/run_bf_e200.sh > outputs/logs/bf_e200.log 2>&1 &
set -uo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg          # stale DISPLAY=:0 hangs matplotlib forever
PY=.venv/bin/python

TAG=e200
EPOCHS=200
STEPS=100
BATCH=256
SIM_WORKERS=20                 # dataset worker PROCESSES, each 1 numba thread
VAL_BATCH=200
VAL_SAMPLES=200

CASES=(intercept_only regressed)

log() { echo "[$(date '+%F %T')] $*"; }

# One model on one GPU, its own CPU half. The taskset split matters: 2 jobs x 20 sim
# workers = 40 processes on 48 cores, and without pinning they fight over the same
# cores and both slow down.
run_model() {
    local model=$1 gpu=$2 cpus=$3
    for case in "${CASES[@]}"; do
        local tlog="outputs/logs/bf_${TAG}_${model}_${case}_train.log"
        local vlog="outputs/logs/bf_${TAG}_${model}_${case}_val.log"

        log "[$model/$case] train ${EPOCHS}ep -> $tlog"
        # NUMBA_THREADING_LAYER=workqueue: train forks its sim workers and the default
        # OpenMP layer is fork-unsafe (silent futex hang, ~8.7GB GPU is the tell).
        taskset -c "$cpus" env CUDA_VISIBLE_DEVICES="$gpu" \
            XLA_PYTHON_CLIENT_PREALLOCATE=false \
            NUMBA_THREADING_LAYER=workqueue \
            "$PY" -m experiments.model_family.model_family_bf_train \
            --model "$model" --case "$case" \
            --epochs "$EPOCHS" --steps_per_epoch "$STEPS" --batch_size "$BATCH" \
            --sim_workers "$SIM_WORKERS" \
            --slice_active_target --time_power_law_alpha 0.0 \
            --no-standardize_summaries \
            --tag "$TAG" > "$tlog" 2>&1
        if [ $? -ne 0 ]; then
            log "[$model/$case] TRAIN FAILED -- see $tlog; next case"
            continue
        fi
        log "[$model/$case] train done"

        # No workqueue here: validate is single-threaded now but does not need it.
        # VAL_BATCH stays at 200 -- this is a shared GPU box and validation is the
        # memory spike, so it is capped rather than raised.
        log "[$model/$case] validate -> $vlog"
        taskset -c "$cpus" env CUDA_VISIBLE_DEVICES="$gpu" \
            XLA_PYTHON_CLIENT_PREALLOCATE=false \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            "$PY" -m experiments.model_family.model_family_bf_validate \
            --model "$model" --case "$case" \
            --batch_size "$VAL_BATCH" --num_samples "$VAL_SAMPLES" \
            --skip_posteriors --tag "$TAG" > "$vlog" 2>&1 \
            && log "[$model/$case] validate done" \
            || log "[$model/$case] VALIDATE FAILED -- see $vlog"
    done
    log "[$model] all cases done"
}

log "=== BF ${TAG}: ${EPOCHS}ep batch ${BATCH}, cases ${CASES[*]} ==="
run_model ddm 0 0-23  &
P0=$!
run_model cdm 1 24-47 &
P1=$!
wait $P0; wait $P1
log "=== BF ${TAG} complete ==="
