#!/usr/bin/env bash
# model_family/run.sh — ModelFamily CF train/validate, BF train/validate, C2ST.
#
# Environment variables (optional):
#   DRY_EPOCHS        — pass --dry_run_epochs to CF training scripts
#   MF_BF_DRY_EPOCHS  — pass --dry_run_epochs to BF training scripts
#   USE_WANDB         — enable wandb logging

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../shared_config.sh"

# How many GPUs are visible to this job?
# Respects CUDA_VISIBLE_DEVICES if already set by a parent script.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c '[0-9]' || true)
    [ "${N_GPUS:-0}" -eq 0 ] && N_GPUS=1
else
    N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)
fi

# NCCL: disable P2P to avoid hang on systems where NVLink/PCIe P2P init stalls;
# falls back to shared memory which works fine for intra-node DDP.
export NCCL_P2P_DISABLE=1

CF_DRY=()
[[ -n "${DRY_EPOCHS:-}" ]] && CF_DRY=(--dry_run_epochs "$DRY_EPOCHS")

BF_DRY=()
[[ -n "${MF_BF_DRY_EPOCHS:-}" ]] && BF_DRY=(--dry_run_epochs "$MF_BF_DRY_EPOCHS")

WANDB=()
[[ -n "${USE_WANDB:-}" ]] && WANDB=(--use_wandb)

SIM_WORKERS_ARG=()
[[ -n "${NUM_SIM_WORKERS:-}" ]] && SIM_WORKERS_ARG=(--num_sim_workers "$NUM_SIM_WORKERS")

cf_train() {
    python -m experiments.model_family.model_family_cf_train \
        --model "$1" \
        --encoder_num_layers "$ENC_LAYERS" \
        --decoder_num_layers "$DEC_LAYERS" \
        --encoder_num_heads  "$HEADS" \
        --decoder_num_heads  "$HEADS" \
        --proj_dim   "$PROJ" \
        --num_seeds  "$SEEDS" \
        --seed_dim   "$DIM" \
        --num_obs    "$OBS" \
        --min_num_obs "$MIN_OBS" \
        --max_num_obs "$MAX_OBS" \
        --train_batch_size "$BATCH" \
        --val_batch_size   "$VAL_BATCH" \
        --epochs          "$EPOCHS" \
        --steps_per_epoch "$STEPS" \
        --lr 1e-4 --dropout 0.05 --layer_dropout 0.05 \
        --fm_sample_steps 50 --fm_num_samples 200 --fm_solver midpoint \
        "${CF_DRY[@]}" "${WANDB[@]}" "${SIM_WORKERS_ARG[@]}"
}

cf_validate() {
    python -m experiments.model_family.model_family_cf_validate \
        --model "$1" --checkpoint "$2" \
        --encoder_num_layers "$ENC_LAYERS" \
        --decoder_num_layers "$DEC_LAYERS" \
        --encoder_num_heads  "$HEADS" \
        --decoder_num_heads  "$HEADS" \
        --proj_dim  "$PROJ" \
        --num_seeds "$SEEDS" \
        --seed_dim  "$DIM" \
        --batch_size 500 --num_samples 200 --sample_chunk_size 50 --samples_chunk_size 100 --num_obs "$OBS"
}

declare -A MF_CKPTS=([ddm]="$DDM_MF_CKPT" [cdm]="$CDM_MF_CKPT" [rdm]="$RDM_MF_CKPT")

# ── CF train → validate per model (sequential DDP, each uses all GPUs) ────────
echo "=== ModelFamily: CF (N_GPUS=${N_GPUS}, DDP) ==="
for model in ddm cdm rdm; do
    echo "  $model: train ..."
    cf_train "$model"
    echo "  $model: validate ..."
    cf_validate "$model" "${MF_CKPTS[$model]}"
done

# ── BF train → validate per model ─────────────────────────────────────────────
echo "=== ModelFamily: BF ==="
for model in ddm cdm rdm; do
    for case in intercept_only fixed regressed; do
        echo "  $model / $case: train ..."
        python -m experiments.model_family.model_family_bf_train \
            --model "$model" --case "$case" \
            --epochs "$MF_BF_EPOCHS" --steps_per_epoch 100 --batch_size 64 \
            "${BF_DRY[@]}" "${WANDB[@]}"
    done
    echo "  $model: validate ..."
    python -m experiments.model_family.model_family_bf_validate --model "$model"
done

# ── C2ST ───────────────────────────────────────────────────────────────────────
echo "=== ModelFamily: C2ST ==="
for model in ddm cdm rdm; do
    python -m experiments.model_family.model_family_c2st --model "$model"
done
