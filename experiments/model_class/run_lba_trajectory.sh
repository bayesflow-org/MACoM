#!/usr/bin/env bash
# Re-run the LBA finetune, KEEPING epoch-stamped checkpoints so the finetuning
# trajectory can be evaluated afterwards.
#
# WHY A FULL RE-RUN: the 2026-08-14 finetune saved every 25 epochs to a FIXED
# filename, so each save destroyed the previous one. Only epoch 2000 survives, and
# a curve over finetuning cannot be reconstructed from it. --trajectory_interval
# keeps an epoch-stamped copy alongside the rolling save.
#
# BATCH=256 IS LOAD-BEARING. run_lba.sh passes --train_batch_size "$MC_BATCH",
# which defaults to 64, while LBA_FT_BATCH in the checkpoint NAME is hardcoded 256.
# wandb confirms the original run used 256. Without exporting BATCH=256 this would
# train at 64 and write to a filename claiming 256.
#
# --seed makes this reproducible; the original run set no seed, which is why its
# trajectory could not be recovered any other way.
#
# GEOMETRY IS ALSO LOAD-BEARING and run_lba.sh omits it: the script's defaults are
# r5c5 (encoder input 186) while the model-class checkpoint is r4c4 (72), so a
# finetune launched through run_lba.sh dies in load_state_dict. wandb confirms the
# 2026-08-14 run passed --max_num_regressors 4 --max_num_categories 4 explicitly,
# i.e. it was NOT launched via run_lba.sh.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source experiments/shared_config.sh
# MC_BATCH is set by shared_config ONLY inside its `if BATCH is unset` branch, so
# exporting BATCH beforehand skips the assignment and leaves MC_BATCH unbound.
# Override after sourcing instead.
MC_BATCH=256

SEED="${SEED:-0}"
TRAJ="${TRAJ:-500}"
LOG="${LOG:-logs/restyle/lba_ft_trajectory.log}"
mkdir -p "$(dirname "$LOG")"

echo "=== LBA FINETUNE (trajectory) start $(date -Is) seed=$SEED traj=$TRAJ batch=$MC_BATCH ===" | tee -a "$LOG"
.venv/bin/python -m experiments.model_class.model_class_cf_finetune_lba \
    --lba_fixed_A "$LBA_FIXED_A" \
    --self_attn_padding_mask \
    --max_num_regressors "$MC_MAX_REG" --max_num_categories "$MC_MAX_CAT" \
    --pretrained_checkpoint "$MC_CKPT" \
    --encoder_num_layers "$ENC_LAYERS" --decoder_num_layers "$DEC_LAYERS" \
    --encoder_num_heads "$HEADS" --decoder_num_heads "$HEADS" \
    --proj_dim "$PROJ" --num_seeds "$SEEDS" --seed_dim "$MC_DIM" \
    --model_embed_dim 8 \
    --epochs "$LBA_FT_EPOCHS" --steps_per_epoch "$STEPS" \
    --lr 1e-5 --dropout 0.05 --layer_dropout 0.05 \
    --train_batch_size "$MC_BATCH" \
    --val_batch_size 50 --val_samples_chunk_size 1 \
    --finetune_scheme 2 \
    --trajectory_interval "$TRAJ" --seed "$SEED" >>"$LOG" 2>&1
echo "=== LBA FINETUNE exit=$? $(date -Is) ===" | tee -a "$LOG"
