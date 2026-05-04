#!/usr/bin/env bash
# generalization/run.sh — Generalization CF train + validate (DDM × 4 holdout types).
#
# Environment variables (optional):
#   DRY_EPOCHS  — pass --dry_run_epochs to training
#   USE_WANDB   — enable wandb logging

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../shared_config.sh"

DRY=()
[[ -n "${DRY_EPOCHS:-}" ]] && DRY=(--dry_run_epochs "$DRY_EPOCHS")

WANDB=()
[[ -n "${USE_WANDB:-}" ]] && WANDB=(--use_wandb)

GEN_CKPT_PREFIX="./outputs/checkpoints/generalization/cf/ddm/cogformer_mixed_attn"

# ── CF train ───────────────────────────────────────────────────────────────────
echo "=== Generalization: CF train ==="
for holdout in random variability interaction sparsity; do
    echo "  ddm / $holdout ..."
    python -m experiments.generalization.generalization_cf_train \
        --model ddm --holdout_type "$holdout" \
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
        --val_epoch_interval 1000 \
        --lr 1e-4 --dropout 0.05 --layer_dropout 0.05 \
        --fm_sample_steps 50 --fm_num_samples 200 --fm_solver midpoint \
        "${DRY[@]}" "${WANDB[@]}"
done

# ── CF validate ────────────────────────────────────────────────────────────────
echo "=== Generalization: CF validate ==="
for holdout in random variability interaction sparsity; do
    gen_ckpt="${GEN_CKPT_PREFIX}_gen_${holdout}_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${DIM}_o${OBS}_b${BATCH}_e${EPOCHS}_t${STEPS}.pt"
    python -m experiments.generalization.generalization_cf_validate \
        --model ddm --holdout_type "$holdout" \
        --checkpoint "$gen_ckpt" \
        --encoder_num_layers "$ENC_LAYERS" \
        --decoder_num_layers "$DEC_LAYERS" \
        --encoder_num_heads  "$HEADS" \
        --decoder_num_heads  "$HEADS" \
        --proj_dim  "$PROJ" \
        --num_seeds "$SEEDS" \
        --seed_dim  "$DIM" \
        --batch_size 500 --num_samples 500 --num_obs "$OBS"
done
