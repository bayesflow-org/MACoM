#!/usr/bin/env bash
# model_class/run.sh — ModelClass CF train/validate + LBA validate.
#
# Environment variables (optional):
#   DRY_EPOCHS  — pass --dry_run_epochs to training
#   USE_WANDB   — enable wandb logging

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../shared_config.sh"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NCCL: disable P2P to avoid hang on systems where NVLink/PCIe P2P init stalls
export NCCL_P2P_DISABLE=1

DRY=()
[[ -n "${DRY_EPOCHS:-}" ]] && DRY=(--dry_run_epochs "$DRY_EPOCHS")

WANDB=()
[[ -n "${USE_WANDB:-}" ]] && WANDB=(--use_wandb)

SIM_WORKERS_ARG=()
[[ -n "${NUM_SIM_WORKERS:-}" ]] && SIM_WORKERS_ARG=(--num_sim_workers "$NUM_SIM_WORKERS")

# ── CF train ───────────────────────────────────────────────────────────────────
echo "=== ModelClass: CF train ==="
python -m experiments.model_class.model_class_cf_train \
    --encoder_num_layers "$ENC_LAYERS" \
    --decoder_num_layers "$DEC_LAYERS" \
    --encoder_num_heads  "$HEADS" \
    --decoder_num_heads  "$HEADS" \
    --proj_dim      "$PROJ" \
    --num_seeds     "$SEEDS" \
    --seed_dim      "$MC_DIM" \
    --model_embed_dim 8 \
    --min_num_obs   "$MIN_OBS" \
    --max_num_obs   "$MAX_OBS" \
    --train_batch_size "$MC_BATCH" \
    --val_batch_size   50 \
    --val_samples_chunk_size 1 \
    --epochs          "$EPOCHS" \
    --steps_per_epoch "$STEPS" \
    --lr 1e-4 --dropout 0.05 --layer_dropout 0.05 \
    --fm_sample_steps 50 --fm_num_samples 100 \
    "${DRY[@]}" "${WANDB[@]}" "${SIM_WORKERS_ARG[@]}"

# ── CF validate ────────────────────────────────────────────────────────────────
echo "=== ModelClass: CF validate ==="
python -m experiments.model_class.model_class_cf_validate \
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

# ── LBA validate (zero-shot) ────────────────────────────────────────────────────
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

# ── LBA fine-tune (Scheme 2 default; override with FINETUNE_SCHEME=1) ──────────
echo "=== ModelClass: LBA fine-tune ==="
FT_SCHEME="${FINETUNE_SCHEME:-2}"
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

# ── LBA fine-tune validate ─────────────────────────────────────────────────────
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

# ── LBA comparison (zero-shot vs. fine-tuned + forgetting) ─────────────────────
echo "=== ModelClass: LBA comparison ==="
python -m experiments.model_class.model_class_lba_compare \
    --zero_shot_lba_pred_dir "./outputs/data/generalization/model_class_lba/cf/" \
    --base_mc_pred_dir       "./outputs/data/model_class/" \
    --finetuned_pred_dir     "./outputs/data/model_class/cf/lba_finetune/" \
    --outdir                 "./outputs/figures/model_class/cf/lba_comparison/"
