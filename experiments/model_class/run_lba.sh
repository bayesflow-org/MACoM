#!/usr/bin/env bash
# run_lba.sh — LBA zero-shot validate, fine-tune, fine-tune validate, and comparison.
#
# Assumes the base MC checkpoint already exists (run run.sh first for training).
#
# Environment variables (optional):
#   FINETUNE_SCHEME  — 1 (LBA-only) or 2 (joint DDM+RDM+CDM+LBA, default)
#   USE_WANDB        — enable wandb logging for fine-tuning
#   SKIP_ZERO_SHOT   — skip the zero-shot LBA validate step
#   SKIP_FINETUNE    — skip fine-tuning (use existing fine-tune checkpoint)
#   SKIP_FT_VALIDATE — skip fine-tune validate (use existing predictions)

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../shared_config.sh"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WANDB=()
[[ -n "${USE_WANDB:-}" ]] && WANDB=(--use_wandb)

FT_SCHEME="${FINETUNE_SCHEME:-2}"

# ── LBA validate (zero-shot) ────────────────────────────────────────────────────
if [[ -z "${SKIP_ZERO_SHOT:-}" ]]; then
    echo "=== ModelClass: LBA validate (zero-shot) ==="
    python -m experiments.generalization.model_class_lba_cf_validate \
        --checkpoint "$MC_CKPT" \
        --encoder_num_layers "$ENC_LAYERS" \
        --decoder_num_layers "$DEC_LAYERS" \
        --encoder_num_heads  "$HEADS" \
        --decoder_num_heads  "$HEADS" \
        --proj_dim      "$PROJ" \
        --num_seeds     "$SEEDS" \
        --seed_dim      "$MC_DIM" \
        --model_embed_dim 8 \
        --batch_size 200 --num_samples 200 --num_obs "$OBS" \
        --samples_chunk_size 1
fi

# ── LBA fine-tune ───────────────────────────────────────────────────────────────
if [[ -z "${SKIP_FINETUNE:-}" ]]; then
    echo "=== ModelClass: LBA fine-tune (scheme ${FT_SCHEME}) ==="
    python -m experiments.model_class.model_class_cf_finetune_lba \
        --pretrained_checkpoint "$MC_CKPT" \
        --encoder_num_layers "$ENC_LAYERS" \
        --decoder_num_layers "$DEC_LAYERS" \
        --encoder_num_heads  "$HEADS" \
        --decoder_num_heads  "$HEADS" \
        --proj_dim      "$PROJ" \
        --num_seeds     "$SEEDS" \
        --seed_dim      "$MC_DIM" \
        --model_embed_dim 8 \
        --epochs "$LBA_FT_EPOCHS" \
        --steps_per_epoch "$STEPS" \
        --lr 1e-5 --dropout 0.05 --layer_dropout 0.05 \
        --train_batch_size "$MC_BATCH" \
        --val_batch_size 50 --val_samples_chunk_size 1 \
        --finetune_scheme "$FT_SCHEME" \
        "${WANDB[@]}"
fi

# ── LBA fine-tune validate ──────────────────────────────────────────────────────
if [[ -z "${SKIP_FT_VALIDATE:-}" ]]; then
    echo "=== ModelClass: LBA fine-tune validate ==="
    python -m experiments.model_class.model_class_lba_finetune_validate \
        --checkpoint "$LBA_FT_CKPT" \
        --encoder_num_layers "$ENC_LAYERS" \
        --decoder_num_layers "$DEC_LAYERS" \
        --encoder_num_heads  "$HEADS" \
        --decoder_num_heads  "$HEADS" \
        --proj_dim      "$PROJ" \
        --num_seeds     "$SEEDS" \
        --seed_dim      "$MC_DIM" \
        --model_embed_dim 8 \
        --batch_size 200 --num_samples 200 --num_obs "$OBS" \
        --samples_chunk_size 1
fi

# ── LBA comparison ──────────────────────────────────────────────────────────────
echo "=== ModelClass: LBA comparison ==="
python -m experiments.model_class.model_class_lba_compare \
    --zero_shot_lba_pred_dir "./outputs/data/generalization/model_class_lba/cf/" \
    --base_mc_pred_dir       "./outputs/data/model_class/" \
    --finetuned_pred_dir     "./outputs/data/model_class/cf/lba_finetune/" \
    --outdir                 "./outputs/figures/model_class/cf/lba_comparison/"
