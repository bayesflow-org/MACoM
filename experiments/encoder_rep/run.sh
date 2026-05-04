#!/usr/bin/env bash
# encoder_rep/run.sh — Encoder Rep CF train, then LCA/FFI BF apply (train + validate).
#
# Environment variables (optional):
#   DRY_EPOCHS          — pass --dry_run_epochs to the CF training script
#   LCA_FFI_DRY_EPOCHS  — pass --dry_run_epochs to LCA/FFI BF training
#   USE_WANDB           — enable wandb logging

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../shared_config.sh"

CF_DRY=()
[[ -n "${DRY_EPOCHS:-}" ]] && CF_DRY=(--dry_run_epochs "$DRY_EPOCHS")

BF_DRY=()
[[ -n "${LCA_FFI_DRY_EPOCHS:-}" ]] && BF_DRY=(--dry_run_epochs "$LCA_FFI_DRY_EPOCHS")

WANDB=()
[[ -n "${USE_WANDB:-}" ]] && WANDB=(--use_wandb)

SHARED_ARCH=(
    --encoder_num_layers "$ENC_LAYERS"
    --decoder_num_layers "$DEC_LAYERS"
    --encoder_num_heads  "$HEADS"
    --decoder_num_heads  "$HEADS"
    --proj_dim           "$PROJ"
    --num_seeds          "$SEEDS"
    --seed_dim           "$MC_DIM"
    --model_embed_dim    8
)

# ── CF train ───────────────────────────────────────────────────────────────────
echo "=== Encoder Rep: CF train ==="
python -m experiments.encoder_rep.encoder_rep_cf_train \
    "${SHARED_ARCH[@]}" \
    --min_num_obs   "$MIN_OBS" \
    --max_num_obs   "$MAX_OBS" \
    --train_batch_size "$BATCH" \
    --val_batch_size   "$VAL_BATCH" \
    --epochs          "$EPOCHS" \
    --steps_per_epoch "$STEPS" \
    --lr 1e-4 --dropout 0.05 --layer_dropout 0.05 \
    --fm_sample_steps 50 --fm_num_samples 100 \
    "${CF_DRY[@]}" "${WANDB[@]}"

# ── LCA / FFI BF apply ─────────────────────────────────────────────────────────
echo "=== Encoder Rep: LCA/FFI BF apply ==="
for case in intercept_only intercept_v; do
    echo "  LCA BF $case — train ..."
    python -m experiments.encoder_rep.lca_bf_apply \
        --mode train --case "$case" \
        --cogformer_checkpoint "$ENC_REP_CKPT" \
        "${SHARED_ARCH[@]}" \
        --epochs "$LCA_FFI_EPOCHS" --steps_per_epoch 100 \
        "${BF_DRY[@]}"

    echo "  LCA BF $case — validate ..."
    python -m experiments.encoder_rep.lca_bf_apply \
        --mode validate --case "$case" \
        --cogformer_checkpoint "$ENC_REP_CKPT" \
        "${SHARED_ARCH[@]}"

    echo "  FFI BF $case — train ..."
    python -m experiments.encoder_rep.ffi_bf_apply \
        --mode train --case "$case" \
        --cogformer_checkpoint "$ENC_REP_CKPT" \
        "${SHARED_ARCH[@]}" \
        --epochs "$LCA_FFI_EPOCHS" --steps_per_epoch 100 \
        "${BF_DRY[@]}"

    echo "  FFI BF $case — validate ..."
    python -m experiments.encoder_rep.ffi_bf_apply \
        --mode validate --case "$case" \
        --cogformer_checkpoint "$ENC_REP_CKPT" \
        "${SHARED_ARCH[@]}"
done
