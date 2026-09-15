#!/usr/bin/env bash
# FULL-BUDGET BayesFlow baselines under the 2026-09-10 pipeline changes.
#   3 models (ddm, cdm, rdm) x 5 cases x 2000 epochs x batch 256, tag __e2000new
#
# WHAT THIS RUN IS FOR. Every 2000-epoch number we have came from the OLD config:
# run_bf_full.sh did not pass --no-standardize_summaries, so production trained with
# summary standardization ON and the library-default flow-matching time exponent until
# --time_power_law_alpha 0.0 was added. The measured epoch/calibration degradation
#
#     ddm/intercept_only  100 ep 0.119 -> 200 ep 0.259     (new config)
#     ddm/intercept_only 2000 ep 0.422                     (OLD config)
#
# therefore mixes two variables at the 2000-epoch end. This run closes that gap: same
# 2000-epoch budget, corrected config, so the epoch curve is measured end to end under
# one configuration.
#
# COST. ~20 s/epoch measured at batch 256 / 100 steps => ~11 h per case, 15 cases,
# 2 GPUs => ~3.5 days wall. Results land incrementally: each case is validated as soon
# as it trains, so outputs/evaluations/.../<model>_families_bf_<case>__e2000new_*.csv
# appears roughly every 11 h rather than all at the end.
#
# RESUMABLE. A case whose evaluations CSV already exists is skipped, so re-launching
# after a crash or a reboot costs nothing. bayesflow checkpoints with save_freq="epoch",
# so an interrupted case also leaves a usable (if short-trained) model.keras behind --
# but this script does NOT resume mid-case, it retrains that case from scratch. Delete
# the evaluations CSV to force a redo.
#
# NOTHING PRODUCTION IS TOUCHED: everything is written under the __e2000new tag.
#
# Usage: setsid nohup bash outputs/run_bf_e2000new.sh > outputs/logs/bf_e2000new.log 2>&1 &
#        DRY_RUN=1 bash outputs/run_bf_e2000new.sh     # print the plan, run nothing
set -uo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
PY=.venv/bin/python

TAG=e2000new
EPOCHS="${BF_EPOCHS:-2000}"
STEPS=100
BATCH=256
SIM_WORKERS=20
VAL_BATCH=200
VAL_SAMPLES=200
DRY="${DRY_RUN:-}"

MODELS=(ddm cdm rdm)
CASES=(intercept_only regressed fixed fixed_regressed interaction)

# Job order is CASE-MAJOR so each case completes across all three models before the
# next case starts. The first six jobs are intercept_only and regressed, which are the
# two cases with existing 100/200/300-epoch reference points -- so the comparison that
# motivated this run is answerable after ~1 day instead of ~3.5.
JOBS=()
for c in "${CASES[@]}"; do for m in "${MODELS[@]}"; do JOBS+=("$m $c"); done; done
NJOBS=${#JOBS[@]}

CTR=outputs/.bf_${TAG}_counter
LOCKF=outputs/.bf_${TAG}_lock
: > "$LOCKF"; echo 0 > "$CTR"

log() { echo "[$(date '+%F %T')] $*"; }

# Atomically hand out the next job index. A shared counter rather than a static
# per-GPU split: cases differ in cost (interaction has 19 active params against
# intercept_only's 7), so a fixed split would leave one card idle for hours.
claim() {
    local i
    exec 200>"$LOCKF"
    flock 200
    i=$(cat "$CTR")
    echo $((i + 1)) > "$CTR"
    flock -u 200
    exec 200>&-
    echo "$i"
}

done_marker() {  # model case -> path whose existence means "already finished"
    echo "outputs/evaluations/model_family/bf/${1}_families_bf_${2}__${TAG}_evaluations.csv"
}

worker() {
    local gpu=$1 cpus=$2 idx job model case
    while true; do
        idx=$(claim)
        [ "$idx" -ge "$NJOBS" ] && break
        job="${JOBS[$idx]}"; model="${job% *}"; case="${job#* }"

        if [ -f "$(done_marker "$model" "$case")" ]; then
            log "[gpu$gpu] ($((idx+1))/$NJOBS) $model/$case already done -- skip"
            continue
        fi

        local tlog="outputs/logs/bf_${TAG}_${model}_${case}_train.log"
        local vlog="outputs/logs/bf_${TAG}_${model}_${case}_val.log"
        log "[gpu$gpu] ($((idx+1))/$NJOBS) $model/$case train ${EPOCHS}ep -> $tlog"

        if [ -n "$DRY" ]; then
            echo "      + train $model/$case on gpu$gpu (cpus $cpus)"
            echo "      + validate $model/$case on gpu$gpu"
            continue
        fi

        # NUMBA_THREADING_LAYER=workqueue: train forks sim workers and the default
        # OpenMP layer is fork-unsafe (silent futex hang; ~8.7 GB GPU is the tell).
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
            log "[gpu$gpu] $model/$case TRAIN FAILED -- see $tlog; next job"
            continue
        fi
        log "[gpu$gpu] $model/$case train done"

        # No workqueue: validate is single-threaded (num_workers=1) and does not need it.
        taskset -c "$cpus" env CUDA_VISIBLE_DEVICES="$gpu" \
            XLA_PYTHON_CLIENT_PREALLOCATE=false \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            "$PY" -m experiments.model_family.model_family_bf_validate \
            --model "$model" --case "$case" \
            --batch_size "$VAL_BATCH" --num_samples "$VAL_SAMPLES" \
            --skip_posteriors --tag "$TAG" > "$vlog" 2>&1 \
            && log "[gpu$gpu] $model/$case validate done" \
            || log "[gpu$gpu] $model/$case VALIDATE FAILED -- see $vlog"
    done
    log "[gpu$gpu] worker exiting -- queue drained"
}

log "=== BF ${TAG}: ${EPOCHS}ep batch ${BATCH}, ${NJOBS} jobs across 2 GPUs ==="
log "order: ${JOBS[*]}"
worker 0 0-23  &
W0=$!
worker 1 24-47 &
W1=$!
wait $W0; wait $W1
log "=== BF ${TAG} complete ==="
