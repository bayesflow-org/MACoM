#!/usr/bin/env bash
# run_bf_full.sh — retrain every BayesFlow baseline and refresh the amortization
# gap after EACH case, so results accrue incrementally instead of at the end.
#
# CASE ORDER (as specified, not alphabetical and not the old bench5 order):
#     intercept_only -> interaction -> fixed -> regressed -> fixed_regressed
#
# PER CASE, per model:
#     1. train            model_family_bf_train           (~10h)
#     2. bf_validate      BF posterior draws -> npz       (~1m)
#     3. cf_validate      PAIRED CogFormer arm            (~1m)
#     4. c2st             rebuild that model's summary    (~1m)
#     5. finalize         refresh outputs/finals/experiment-1
#
# WHY THE EVAL IS PER CASE. bf_validate is seeded per (model, case) as of
# 2026-09-10, so each case's npz is reproducible and pairing is stable. c2st and
# the finalizer both skip what is absent, so running them after every case simply
# grows the summary a row at a time. A case that fails leaves the earlier rows
# intact.
#
# WHY --data_dir POINTS AT A ONE-CASE DIRECTORY. model_class_cf_validate has no
# --case filter; in paired mode it processes every case whose npz it can find. A
# directory holding a single symlink restricts it to the case just trained,
# instead of resampling every finished case each time.
#
# FIXES CARRIED BY THIS RUN (all measured, see the A/B logs):
#   --slice_active_target   train the flow on ACTIVE cells only. MARGINAL, not a
#                           fix: measured -3% calibration error (0.417 -> 0.405),
#                           and the predecessor project fed the full grid and
#                           still reached 0.019. Kept because it is free.
#   --time_power_law_alpha 0.0
#                           uniform flow-matching time, matching CogFormer's
#                           decoder (t = torch.rand). BayesFlow defaults to 0.5,
#                           which starves the data end where the field learns to
#                           contract. Measured worth ~11% calibration error.
#   NUMBA_THREADING_LAYER=workqueue   fork-safe sim workers (else silent deadlock)
#   MPLBACKEND=Agg                    stale DISPLAY=:0 hangs matplotlib forever
#
# LBA IS DIFFERENT. Its gap cell is BF_vs_Finetuned, not BF_vs_Original: it has no
# original-checkpoint arm. Its CogFormer side comes from
# model_class_lba_finetune_validate, and the finalizer deliberately excludes LBA
# (not one of Experiment 1's three families). That path has never been run, so its
# failures are logged and skipped rather than allowed to block training.
#
# Usage: setsid nohup bash outputs/run_bf_full.sh > outputs/logs/bf_full.log 2>&1 &
set -uo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# NUMBA_THREADING_LAYER is deliberately NOT exported globally: the two stages need
# OPPOSITE layers, and getting it wrong breaks one of them silently.
#
#   train      forks its sim workers (OnlineDataset use_multiprocessing=True) and
#              runs ModelFamily with num_workers=1, i.e. no threads. The default
#              OpenMP-backed layer is fork-UNSAFE -> children die with "fork()
#              called from a process already using GNU OpenMP" and the parent
#              hangs on a futex forever. Needs workqueue.
#
#   bf_validate  does NOT fork, but constructs ModelFamily WITHOUT num_workers=1,
#              so it simulates on ~8 threads. workqueue is not threadsafe ->
#              "Numba workqueue threading layer is terminating: Concurrent access
#              has been detected" and the process dies. Needs the default layer.
#
# Caught by a DRY_RUN walkthrough on 2026-09-10, where a globally exported
# workqueue killed bf_validate before it wrote anything. So workqueue is set on
# the TRAIN invocation only, below.
export MPLBACKEND=Agg
PY=.venv/bin/python
source experiments/shared_config.sh

CASES=(intercept_only interaction fixed regressed fixed_regressed)
# NOT ${EPOCHS:-2000}: shared_config.sh is sourced above and sets EPOCHS=10000
# (the CogFormer model-class budget), so the :- default never fired and every
# case silently ran 5x too long -- 21 days projected instead of 4.3. Read the
# override from a distinct name so a sourced config cannot shadow it.
EPOCHS="${BF_EPOCHS:-2000}"
STEPS=100
BATCH=256
SIM_WORKERS=20

BF_DATA_DIR="./outputs/data/model_family"
PRED_DIR="./outputs/data/model_family_paired"
FT_PRED_DIR="./outputs/data/model_family_paired_ft"
FIG_DIR="./outputs/figures/model_class/cf_paired/"
FT_FIG_DIR="./outputs/figures/model_class/cf_paired_ft/"
SPLIT_ROOT="./outputs/data/_mf_onecase"
LOCK="./outputs/.finalize.lock"

mkdir -p "$BF_DATA_DIR" "$PRED_DIR" "$FT_PRED_DIR" "$SPLIT_ROOT"
touch "$LOCK"

log() { echo "[$(date '+%F %T')] $*"; }

# DRY_RUN=1 prints the pipeline without running anything: verifies case order,
# wave assignment and argv construction before committing ~10 GPU-days.
DRY="${DRY_RUN:-}"
maybe() { if [ -n "$DRY" ]; then echo "      + $*" | cut -c1-150; return 0; fi; "$@"; }

# ── evaluation for one (model, case) ──────────────────────────────────────────
eval_case() {
    local model=$1 case=$2 gpu=$3 cpus=$4
    local run="taskset -c $cpus env CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

    log "[$model/$case] 2/5 bf_validate"
    maybe $run "$PY" -m experiments.model_family.model_family_bf_validate \
        --model "$model" --case "$case" \
        --batch_size "$VAL_BATCH" --num_samples 200 --skip_posteriors \
        > "outputs/logs/bfval_${model}_${case}.log" 2>&1 \
        || { log "[$model/$case] bf_validate FAILED; skipping rest of eval"; return 1; }

    # One-case view so cf_validate does not resample every finished case.
    local sd="$SPLIT_ROOT/${model}_${case}"
    rm -rf "$sd"; mkdir -p "$sd"
    if ! ln -sf "$(realpath "$BF_DATA_DIR/${model}_${case}_data.npz" 2>/dev/null)" "$sd/" 2>/dev/null; then
        [ -z "$DRY" ] && { log "[$model/$case] npz missing after bf_validate; skipping"; return 1; }
    fi

    if [ "$model" = "lba" ]; then
        log "[$model/$case] 3/5 lba_finetune_validate (BF_vs_Finetuned arm)"
        maybe $run "$PY" -m experiments.model_class.model_class_lba_finetune_validate \
            --checkpoint "$LBA_FT_CKPT" --lba_fixed_A "$LBA_FIXED_A" \
            --self_attn_padding_mask \
            --encoder_num_layers "$ENC_LAYERS" --decoder_num_layers "$DEC_LAYERS" \
            --encoder_num_heads "$HEADS" --decoder_num_heads "$HEADS" \
            --proj_dim "$PROJ" --num_seeds "$SEEDS" --seed_dim "$MC_DIM" \
            --model_embed_dim 8 \
            --max_num_regressors "$MC_MAX_REG" --max_num_categories "$MC_MAX_CAT" \
            --data_dir "$sd" --pred_dir "$FT_PRED_DIR" --outdir "$FT_FIG_DIR" \
            --batch_size "$VAL_BATCH" --num_samples 200 --num_obs "$OBS" \
            --samples_chunk_size 1 \
            > "outputs/logs/ftval_${model}_${case}.log" 2>&1 \
            || { log "[$model/$case] ft_validate FAILED (untested path); training continues"; return 1; }

        log "[$model/$case] 4/5 c2st BF_vs_Finetuned"
        maybe $run "$PY" -m experiments.model_family.model_family_c2st \
            --model "$model" --bf_data_dir "$BF_DATA_DIR" \
            --ft_pred_dir "$FT_PRED_DIR" --scenarios BF_vs_Finetuned \
            --max_num_categories "$MC_MAX_CAT" \
            > "outputs/logs/c2st_${model}.log" 2>&1 \
            || { log "[$model/$case] c2st FAILED"; return 1; }
        log "[$model/$case] 5/5 finalize: LBA is excluded from experiment-1 by design; skipped"
        return 0
    fi

    log "[$model/$case] 3/5 cf_validate (paired)"
    maybe $run "$PY" -m experiments.model_class.model_class_cf_validate \
        --self_attn_padding_mask \
        --encoder_num_layers "$ENC_LAYERS" --decoder_num_layers "$DEC_LAYERS" \
        --encoder_num_heads "$HEADS" --decoder_num_heads "$HEADS" \
        --proj_dim "$PROJ" --num_seeds "$SEEDS" --seed_dim "$MC_DIM" \
        --model_embed_dim 8 \
        --max_num_regressors "$MC_MAX_REG" --max_num_categories "$MC_MAX_CAT" \
        --checkpoint "$MC_CKPT" \
        --data_dir "$sd" --pred_dir "$PRED_DIR" --outdir "$FIG_DIR" \
        --batch_size "$VAL_BATCH" --num_samples 200 --num_obs "$OBS" \
        --samples_chunk_size 1 --skip_posteriors \
        > "outputs/logs/cfval_${model}_${case}.log" 2>&1 \
        || { log "[$model/$case] cf_validate FAILED; skipping rest of eval"; return 1; }

    log "[$model/$case] 4/5 c2st BF_vs_Original"
    maybe $run "$PY" -m experiments.model_family.model_family_c2st \
        --model "$model" --bf_data_dir "$BF_DATA_DIR" \
        --mc_pred_dir "$PRED_DIR" --scenarios BF_vs_Original \
        --max_num_categories "$MC_MAX_CAT" \
        > "outputs/logs/c2st_${model}.log" 2>&1 \
        || { log "[$model/$case] c2st FAILED"; return 1; }

    # Both model slots can reach this at once, and the finalizer rewrites
    # MANIFEST.csv and README.md wholesale -- concurrent runs would interleave.
    log "[$model/$case] 5/5 finalize (lock)"
    maybe flock "$LOCK" "$PY" -m experiments.finalize_experiment_1 \
        > "outputs/logs/finalize_${model}_${case}.log" 2>&1 \
        || log "[$model/$case] finalize FAILED (results still on disk under outputs/figures/c2st)"

    local s="./outputs/figures/c2st/${model}/${model}_c2st_summary.csv"
    [ -f "$s" ] && log "[$model/$case] summary rows: $(( $(wc -l < "$s") - 1 ))"
    return 0
}

# ── one model: every case, train then evaluate ────────────────────────────────
run_model() {
    local model=$1 gpu=$2 cpus=$3
    log "[$model] start on GPU$gpu (cpus $cpus), cases: ${CASES[*]}"
    for case in "${CASES[@]}"; do
        local out="outputs/logs/bf_${model}_${case}_$(date +%Y%m%d_%H%M%S).log"
        log "[$model/$case] 1/5 train -> $out"
        maybe taskset -c "$cpus" env CUDA_VISIBLE_DEVICES="$gpu" \
            XLA_PYTHON_CLIENT_PREALLOCATE=false \
            NUMBA_THREADING_LAYER=workqueue \
            "$PY" -m experiments.model_family.model_family_bf_train \
            --model "$model" --case "$case" \
            --epochs "$EPOCHS" --steps_per_epoch "$STEPS" --batch_size "$BATCH" \
            --sim_workers "$SIM_WORKERS" \
            --slice_active_target --time_power_law_alpha 0.0 \
            --use_wandb > "$out" 2>&1
        local rc=$?
        if [ $rc -ne 0 ]; then
            log "[$model/$case] TRAIN FAILED (exit $rc) -- see $out; skipping eval, next case"
            continue
        fi
        log "[$model/$case] train done"
        eval_case "$model" "$case" "$gpu" "$cpus" \
            && log "[$model/$case] eval complete" \
            || log "[$model/$case] eval incomplete -- see logs above"
    done
    log "[$model] all cases done"
}

WAVE1=(${WAVE1:-ddm cdm})
WAVE2=(${WAVE2:-rdm lba})

log "=== wave 1: ${WAVE1[*]} ==="
run_model "${WAVE1[0]}" 0 0-23  & p1=$!
run_model "${WAVE1[1]}" 1 24-47 & p2=$!
wait $p1 $p2
log "=== wave 1 complete ==="

log "=== wave 2: ${WAVE2[*]} ==="
run_model "${WAVE2[0]}" 0 0-23  & p3=$!
run_model "${WAVE2[1]}" 1 24-47 & p4=$!
wait $p3 $p4
log "=== wave 2 complete ==="

log "=== ALL DONE ==="
for m in "${WAVE1[@]}" "${WAVE2[@]}"; do
    s="./outputs/figures/c2st/${m}/${m}_c2st_summary.csv"
    [ -f "$s" ] && { log "--- $m ---"; cat "$s"; } || log "--- $m: no summary ---"
done
