#!/usr/bin/env bash
# run_experiment.sh — Full experiment pipeline with 2-GPU parallel scheduling.
#
# Phase 1: DDM (GPU 0) + CDM (GPU 1) in parallel
# Phase 2: RDM (GPU 0) + model_class full pipeline (GPU 1) in parallel
# Phase 3: BF trains + C2ST (CPU-bound, sequential)
#
# Environment variables:
#   USE_WANDB=1        — enable wandb logging
#   NUM_SIM_WORKERS=N  — number of parallel simulation workers (default 8)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

source experiments/shared_config.sh

export BATCH=64 MC_BATCH=64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUM_GPU_JOBS=2

WANDB=()
[[ -n "${USE_WANDB:-}" ]] && WANDB=(--use_wandb)
SIM_WORKERS_ARG=()
[[ -n "${NUM_SIM_WORKERS:-}" ]] && SIM_WORKERS_ARG=(--num_sim_workers "$NUM_SIM_WORKERS")

mf_cf_train() {
    local gpu=$1; local model=$2
    CUDA_VISIBLE_DEVICES="$gpu" python -m experiments.model_family.model_family_cf_train \
        --model "$model" \
        --encoder_num_layers "$ENC_LAYERS" --decoder_num_layers "$DEC_LAYERS" \
        --encoder_num_heads  "$HEADS"       --decoder_num_heads  "$HEADS" \
        --proj_dim "$PROJ" --num_seeds "$SEEDS" --seed_dim "$DIM" \
        --num_obs "$OBS" --min_num_obs "$MIN_OBS" --max_num_obs "$MAX_OBS" \
        --train_batch_size "$BATCH" --val_batch_size "$VAL_BATCH" \
        --epochs "$EPOCHS" --steps_per_epoch "$STEPS" \
        --lr 1e-4 --dropout 0.05 --layer_dropout 0.05 \
        --fm_sample_steps 50 --fm_num_samples 200 --fm_solver midpoint \
        "${WANDB[@]}" "${SIM_WORKERS_ARG[@]}"
}

mf_cf_validate() {
    local model=$1; local ckpt=$2
    python -m experiments.model_family.model_family_cf_validate \
        --model "$model" --checkpoint "$ckpt" \
        --encoder_num_layers "$ENC_LAYERS" --decoder_num_layers "$DEC_LAYERS" \
        --encoder_num_heads "$HEADS" --decoder_num_heads "$HEADS" \
        --proj_dim "$PROJ" --num_seeds "$SEEDS" --seed_dim "$DIM" \
        --batch_size 500 --num_samples 200 --sample_chunk_size 50 \
        --samples_chunk_size 100 --num_obs "$OBS"
}

# ── Phase 1: DDM (GPU 0) + CDM (GPU 1) ───────────────────────────────────────
echo "=== Phase 1: DDM (GPU 0) + CDM (GPU 1) ==="
mf_cf_train 0 ddm &
PID_DDM=$!
mf_cf_train 1 cdm &
PID_CDM=$!
wait $PID_DDM $PID_CDM

echo "=== Validate DDM + CDM ==="
mf_cf_validate ddm "$DDM_MF_CKPT"
mf_cf_validate cdm "$CDM_MF_CKPT"

# ── Phase 2: RDM (GPU 0) + model_class (GPU 1) ───────────────────────────────
echo "=== Phase 2: RDM (GPU 0) + model_class (GPU 1) ==="
mf_cf_train 0 rdm &
PID_RDM=$!
CUDA_VISIBLE_DEVICES=1 bash experiments/model_class/run.sh &
PID_MC=$!
wait $PID_RDM $PID_MC

echo "=== Validate RDM ==="
mf_cf_validate rdm "$RDM_MF_CKPT"

# ── Phase 3: BF trains + C2ST (CPU-bound) ────────────────────────────────────
echo "=== Phase 3: BF trains ==="
BF_DRY=()
[[ -n "${MF_BF_DRY_EPOCHS:-}" ]] && BF_DRY=(--dry_run_epochs "$MF_BF_DRY_EPOCHS")
for model in ddm cdm rdm; do
    for case in intercept_only fixed regressed; do
        echo "  $model / $case ..."
        python -m experiments.model_family.model_family_bf_train \
            --model "$model" --case "$case" \
            --epochs "$MF_BF_EPOCHS" --steps_per_epoch 100 --batch_size 64 \
            "${BF_DRY[@]}" "${WANDB[@]}"
    done
    python -m experiments.model_family.model_family_bf_validate --model "$model"
done

echo "=== C2ST ==="
for model in ddm cdm rdm; do
    python -m experiments.model_family.model_family_c2st --model "$model"
done

echo "=== Experiment complete ==="
