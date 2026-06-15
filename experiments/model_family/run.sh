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
# Reduce allocator fragmentation — important on this shared box where a co-tenant
# can hold tens of GB, leaving us a tight, fragmented remainder.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CF_DRY=()
[[ -n "${DRY_EPOCHS:-}" ]] && CF_DRY=(--dry_run_epochs "$DRY_EPOCHS")

BF_DRY=()
[[ -n "${MF_BF_DRY_EPOCHS:-}" ]] && BF_DRY=(--dry_run_epochs "$MF_BF_DRY_EPOCHS")

# USE_WANDB controls CF (CogFormer) logging only. BF (BayesFlow baseline) logging
# is opt-in via BF_USE_WANDB so a "CF-only" run doesn't also log the baselines.
WANDB=()
[[ -n "${USE_WANDB:-}" ]] && WANDB=(--use_wandb)
BF_WANDB=()
[[ -n "${BF_USE_WANDB:-}" ]] && BF_WANDB=(--use_wandb)

SIM_WORKERS_ARG=()
[[ -n "${NUM_SIM_WORKERS:-}" ]] && SIM_WORKERS_ARG=(--num_sim_workers "$NUM_SIM_WORKERS")

# Phase/model selectors (space-separated; empty string skips the phase).
# Defaults run everything, so calling this script with no overrides is unchanged.
# run_all.sh sets these to schedule individual families across GPUs in waves.
CF_MODELS="${CF_MODELS-ddm cdm rdm}"
BF_MODELS="${BF_MODELS-ddm cdm rdm}"
C2ST_MODELS="${C2ST_MODELS-ddm cdm rdm}"

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
        --val_batch_size   "${CF_VAL_BATCH:-50}" \
        --val_samples_chunk_size "${CF_VAL_CHUNK:-1}" \
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

# ── CF train → validate per model (single GPU per job) ────────────────────────
if [ -n "${CF_MODELS// }" ]; then
    echo "=== ModelFamily: CF (N_GPUS=${N_GPUS}) — models: ${CF_MODELS} ==="
    for model in $CF_MODELS; do
        echo "  $model: train ..."
        cf_train "$model"
        echo "  $model: validate ..."
        cf_validate "$model" "${MF_CKPTS[$model]}"
    done
fi

# ── BF train → validate per model ─────────────────────────────────────────────
if [ -n "${BF_MODELS// }" ]; then
    echo "=== ModelFamily: BF — models: ${BF_MODELS} ==="
    for model in $BF_MODELS; do
        for case in intercept_only fixed regressed; do
            echo "  $model / $case: train ..."
            python -m experiments.model_family.model_family_bf_train \
                --model "$model" --case "$case" \
                --epochs "$MF_BF_EPOCHS" --steps_per_epoch 100 --batch_size 64 \
                "${BF_DRY[@]}" "${BF_WANDB[@]}"
        done
        echo "  $model: validate ..."
        python -m experiments.model_family.model_family_bf_validate --model "$model"
    done
fi

# ── C2ST ───────────────────────────────────────────────────────────────────────
if [ -n "${C2ST_MODELS// }" ]; then
    echo "=== ModelFamily: C2ST — models: ${C2ST_MODELS} ==="
    for model in $C2ST_MODELS; do
        python -m experiments.model_family.model_family_c2st --model "$model"
    done
fi
