#!/usr/bin/env bash
# Train, validate, and run ensemble for all CogFormer model-family models.
# Workflow per model: train -> validate -> ensemble
#
# Usage:
#   bash experiments/model_family/run_cf_all.sh
#   bash experiments/model_family/run_cf_all.sh --epochs 2000 --train_batch_size 128
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
SEED_DIM=64
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
NUM_OBS=500
MIN_NUM_OBS=200
MAX_NUM_OBS=500

# Inference
FM_STEPS=200
FM_SAMPLES=200

# ============================================================
# Derived checkpoint path components (must match cf_train naming)
# ============================================================
# Format: {prefix}_l{dec_layers}_h{dec_heads}_p{proj}_s{seeds}_d{seed_dim}
#                 _o{num_obs}_b{train_batch}_e{epochs}_t{steps}.pt

CKPT_SUFFIX="_l${DECODER_LAYERS}_h${DECODER_HEADS}_p${PROJ_DIM}_s${NUM_SEEDS}_d${SEED_DIM}_o${NUM_OBS}_b${TRAIN_BATCH}_e${EPOCHS}_t${STEPS_PER_EPOCH}.pt"

declare -A CKPT_DIR CKPT_PREFIX
CKPT_DIR["ddm"]="./outputs/checkpoints/model_family/cf/ddm"
CKPT_DIR["cdm"]="./outputs/checkpoints/model_family/cf/cdm"
CKPT_DIR["rdm"]="./outputs/checkpoints/model_family/cf/rdm"
CKPT_PREFIX["ddm"]="cogformer_mixed_attn"
CKPT_PREFIX["cdm"]="cogformer_cdm_mixed_attn"
CKPT_PREFIX["rdm"]="cogformer_rdm_mixed_attn"

# ============================================================
# Scripts
# ============================================================
TRAIN_SCRIPT="experiments/model_family/model_family_cf_train.py"
VALIDATE_SCRIPT="experiments/model_family/model_family_cf_validate.py"
ENSEMBLE_SCRIPT="experiments/model_family/model_family_cf_ensemble.py"

# Shared architecture args — reused verbatim across train / validate / ensemble
ARCH_ARGS=(
    --encoder_num_layers "$ENCODER_LAYERS"
    --decoder_num_layers "$DECODER_LAYERS"
    --encoder_num_heads  "$ENCODER_HEADS"
    --decoder_num_heads  "$DECODER_HEADS"
    --projection_dim     "$PROJ_DIM"
    --num_seeds          "$NUM_SEEDS"
    --seed_dim           "$SEED_DIM"
    --time_embedding_dim "$TIME_EMB_DIM"
    --pos_embedding_dim  "$POS_EMB_DIM"
    --dropout            "$DROPOUT"
    --layer_dropout      "$LAYER_DROPOUT"
)

EXTRA_TRAIN_ARGS=("$@")   # passthrough overrides from command line

MODELS=("ddm" "cdm" "rdm")
FAILED=()
PASSED=()

echo "============================================================"
echo " CogFormer Model Family — train / validate / ensemble"
echo " Models : ${MODELS[*]}"
echo " Epochs : $EPOCHS  |  Batch : $TRAIN_BATCH  |  Steps : $STEPS_PER_EPOCH"
echo " Extra train args: ${EXTRA_TRAIN_ARGS[*]:-(none)}"
echo "============================================================"

for model in "${MODELS[@]}"; do

    CKPT_PATH="${CKPT_DIR[$model]}/${CKPT_PREFIX[$model]}${CKPT_SUFFIX}"

    echo ""
    echo "------------------------------------------------------------"
    echo " TRAIN  :  $model"
    echo "------------------------------------------------------------"
    if ! python "$TRAIN_SCRIPT" \
            --model "$model" \
            "${ARCH_ARGS[@]}" \
            --lr              "$LR" \
            --epochs          "$EPOCHS" \
            --steps_per_epoch "$STEPS_PER_EPOCH" \
            --train_batch_size "$TRAIN_BATCH" \
            --val_batch_size  "$VAL_BATCH" \
            --num_obs         "$NUM_OBS" \
            --min_num_obs     "$MIN_NUM_OBS" \
            --max_num_obs     "$MAX_NUM_OBS" \
            --fm_sample_steps "$FM_STEPS" \
            --fm_num_samples  "$FM_SAMPLES" \
            "${EXTRA_TRAIN_ARGS[@]}"; then
        echo "  [FAIL] Training failed — $model"
        FAILED+=("${model}/train")
        continue
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo " VALIDATE  :  $model"
    echo "------------------------------------------------------------"
    if ! python "$VALIDATE_SCRIPT" \
            --model      "$model" \
            --checkpoint "$CKPT_PATH" \
            "${ARCH_ARGS[@]}" \
            --batch_size       "$VAL_BATCH" \
            --num_obs          "$NUM_OBS" \
            --num_sample_steps "$FM_STEPS" \
            --num_samples      "$FM_SAMPLES"; then
        echo "  [FAIL] Validation failed — $model"
        FAILED+=("${model}/validate")
        continue
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo " ENSEMBLE  :  $model"
    echo "------------------------------------------------------------"
    if ! python "$ENSEMBLE_SCRIPT" \
            --model      "$model" \
            --checkpoint "$CKPT_PATH" \
            "${ARCH_ARGS[@]}" \
            --batch_size       "$VAL_BATCH" \
            --num_obs          "$NUM_OBS" \
            --num_sample_steps "$FM_STEPS" \
            --num_samples      "$FM_SAMPLES"; then
        echo "  [FAIL] Ensemble failed — $model"
        FAILED+=("${model}/ensemble")
        continue
    fi

    PASSED+=("$model")

done

echo ""
echo "============================================================"
echo " Summary"
echo "============================================================"
echo " Passed (${#PASSED[@]}) : ${PASSED[*]:-(none)}"
echo " Failed (${#FAILED[@]}) : ${FAILED[*]:-(none)}"
echo "============================================================"
