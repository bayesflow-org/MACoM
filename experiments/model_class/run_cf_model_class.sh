#!/usr/bin/env bash
# Train, validate, and run ensemble for the CogFormer model class.
# Workflow: train -> validate -> ensemble
#
# Usage:
#   bash experiments/model_class/run_cf_model_class.sh
#   bash experiments/model_class/run_cf_model_class.sh --epochs 2000 --train_batch_size 128
#
# Any flags after the script name override the defaults below and are forwarded
# to the training script only. Validate and ensemble always use the centralized
# defaults defined here.
#
# Run from the project root directory.

set -euo pipefail

# ============================================================
# Centralized config
# ============================================================

# Architecture
ENCODER_LAYERS=8
DECODER_LAYERS=8
ENCODER_HEADS=8
DECODER_HEADS=8
PROJ_DIM=256
NUM_SEEDS=32
SEED_DIM=128
MODEL_EMBED_DIM=8
TIME_EMB_DIM=32
POS_EMB_DIM=32
DROPOUT=0.05
LAYER_DROPOUT=0.05

# Training
LR=1e-4
EPOCHS=5000
STEPS_PER_EPOCH=100
TRAIN_BATCH=64
VAL_BATCH=200
MIN_NUM_OBS=200
MAX_NUM_OBS=500
VAL_NUM_OBS=500

# Inference
FM_STEPS=200
FM_SAMPLES=100

# ============================================================
# Derived checkpoint path (must match model_class_cf_train naming)
# ============================================================
# Format: cogformer_model_class_fm_l{dec_layers}_h{dec_heads}_p{proj}
#                                _s{seeds}_d{seed_dim}_b{batch}_e{epochs}_t{steps}.pt

CKPT_DIR="./outputs/checkpoints/model_class/cf"
CKPT_FILE="cogformer_model_class_fm_l${DECODER_LAYERS}_h${DECODER_HEADS}_p${PROJ_DIM}_s${NUM_SEEDS}_d${SEED_DIM}_b${TRAIN_BATCH}_e${EPOCHS}_t${STEPS_PER_EPOCH}.pt"
CKPT_PATH="${CKPT_DIR}/${CKPT_FILE}"

# ============================================================
# Scripts
# ============================================================
TRAIN_SCRIPT="experiments/model_class/model_class_cf_train.py"
VALIDATE_SCRIPT="experiments/model_class/model_class_cf_validate.py"
ENSEMBLE_SCRIPT="experiments/model_class/model_class_cf_ensemble.py"

# Shared architecture args — reused verbatim across train / validate / ensemble
ARCH_ARGS=(
    --encoder_num_layers "$ENCODER_LAYERS"
    --decoder_num_layers "$DECODER_LAYERS"
    --encoder_num_heads  "$ENCODER_HEADS"
    --decoder_num_heads  "$DECODER_HEADS"
    --projection_dim     "$PROJ_DIM"
    --num_seeds          "$NUM_SEEDS"
    --seed_dim           "$SEED_DIM"
    --model_embed_dim    "$MODEL_EMBED_DIM"
    --time_embedding_dim "$TIME_EMB_DIM"
    --pos_embedding_dim  "$POS_EMB_DIM"
    --dropout            "$DROPOUT"
    --layer_dropout      "$LAYER_DROPOUT"
)

EXTRA_TRAIN_ARGS=("$@")

echo "============================================================"
echo " CogFormer Model Class — train / validate / ensemble"
echo " Checkpoint : $CKPT_PATH"
echo " Epochs     : $EPOCHS  |  Batch : $TRAIN_BATCH  |  Steps : $STEPS_PER_EPOCH"
echo " Extra train args: ${EXTRA_TRAIN_ARGS[*]:-(none)}"
echo "============================================================"

echo ""
echo "------------------------------------------------------------"
echo " TRAIN"
echo "------------------------------------------------------------"
python "$TRAIN_SCRIPT" \
    "${ARCH_ARGS[@]}" \
    --lr               "$LR" \
    --epochs           "$EPOCHS" \
    --steps_per_epoch  "$STEPS_PER_EPOCH" \
    --train_batch_size "$TRAIN_BATCH" \
    --val_batch_size   "$VAL_BATCH" \
    --min_num_obs      "$MIN_NUM_OBS" \
    --max_num_obs      "$MAX_NUM_OBS" \
    --val_num_obs      "$VAL_NUM_OBS" \
    --fm_sample_steps  "$FM_STEPS" \
    --fm_num_samples   "$FM_SAMPLES" \
    "${EXTRA_TRAIN_ARGS[@]}"

echo ""
echo "------------------------------------------------------------"
echo " VALIDATE"
echo "------------------------------------------------------------"
python "$VALIDATE_SCRIPT" \
    --checkpoint       "$CKPT_PATH" \
    "${ARCH_ARGS[@]}" \
    --batch_size       "$VAL_BATCH" \
    --num_obs          "$VAL_NUM_OBS" \
    --num_sample_steps "$FM_STEPS" \
    --num_samples      "$FM_SAMPLES"

echo ""
echo "------------------------------------------------------------"
echo " ENSEMBLE"
echo "------------------------------------------------------------"
python "$ENSEMBLE_SCRIPT" \
    --checkpoint       "$CKPT_PATH" \
    "${ARCH_ARGS[@]}" \
    --batch_size       "$VAL_BATCH" \
    --num_obs          "$VAL_NUM_OBS" \
    --num_sample_steps "$FM_STEPS" \
    --num_samples      "$FM_SAMPLES"

echo ""
echo "============================================================"
echo " Done."
echo "============================================================"
