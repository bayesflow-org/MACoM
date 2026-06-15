#!/usr/bin/env bash
# Train, validate, and run ensemble for all CogFormer model-family models.
# Workflow per model: train -> validate -> ensemble
#
# Usage:
#   bash experiments/model_family/run_cf_all.sh
#   bash experiments/model_family/run_cf_all.sh --epochs 1000 --train_batch_size 128
#
# Any flags after the script name override the defaults below and are forwarded
# to the training script only. Validate and ensemble always use the centralized
# defaults defined here.
#
# Run from the project root directory.

set -euo pipefail

# Ensure the project root is on PYTHONPATH so `cogformer` is importable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

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

# Logging
USE_WANDB="${USE_WANDB:-true}"   # set USE_WANDB=false to skip W&B logging

# Training
LR=1e-4
EPOCHS=5000
STEPS_PER_EPOCH=100
TRAIN_BATCH=64
VAL_BATCH=500
NUM_OBS=500
MIN_NUM_OBS=200
MAX_NUM_OBS=1000

# Inference
FM_STEPS=100
FM_SAMPLES=200       # fm_num_samples for training val_step
CF_BATCH=500         # batch_size for cf_validate / ensemble
CF_FM_SAMPLES=500    # num_samples for cf_validate / ensemble
FM_SOLVER="euler"    # euler | midpoint | rk4

# ============================================================
# Apply any command-line overrides to centralized variables.
# This ensures CKPT_SUFFIX matches the checkpoint the training script
# will produce, and that validate/ensemble use the same settings.
# ============================================================
_args=("$@")
for (( _i=0; _i<${#_args[@]}; _i++ )); do
    case "${_args[$_i]}" in
        --epochs)           EPOCHS="${_args[$(( _i+1 ))]}";         _i=$(( _i+1 )) ;;
        --steps_per_epoch)  STEPS_PER_EPOCH="${_args[$(( _i+1 ))]}"; _i=$(( _i+1 )) ;;
        --train_batch_size) TRAIN_BATCH="${_args[$(( _i+1 ))]}";    _i=$(( _i+1 )) ;;
        --val_batch_size)   VAL_BATCH="${_args[$(( _i+1 ))]}";      _i=$(( _i+1 )) ;;
        --num_obs)          NUM_OBS="${_args[$(( _i+1 ))]}";        _i=$(( _i+1 )) ;;
        --fm_sample_steps)  FM_STEPS="${_args[$(( _i+1 ))]}";       _i=$(( _i+1 )) ;;
        --fm_num_samples)   FM_SAMPLES="${_args[$(( _i+1 ))]}";     _i=$(( _i+1 )) ;;
        --fm_solver)        FM_SOLVER="${_args[$(( _i+1 ))]}";      _i=$(( _i+1 )) ;;
    esac
done

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
TRAIN_MODULE="experiments.model_family.model_family_cf_train"
VALIDATE_MODULE="experiments.model_family.model_family_cf_validate"
ENSEMBLE_MODULE="experiments.model_family.model_family_cf_ensemble"

# Shared architecture args — reused verbatim across train / validate / ensemble
ARCH_ARGS=(
    --encoder_num_layers "$ENCODER_LAYERS"
    --decoder_num_layers "$DECODER_LAYERS"
    --encoder_num_heads  "$ENCODER_HEADS"
    --decoder_num_heads  "$DECODER_HEADS"
    --proj_dim           "$PROJ_DIM"
    --num_seeds          "$NUM_SEEDS"
    --seed_dim           "$SEED_DIM"
    --time_embedding_dim "$TIME_EMB_DIM"
    --pos_embedding_dim  "$POS_EMB_DIM"
    --dropout            "$DROPOUT"
    --layer_dropout      "$LAYER_DROPOUT"
)

# Decoder self-attention padding mask. Default on (the architecture default).
# Set SELF_ATTN_PADDING_MASK=false to drop it (~1.48x faster/step per the mask A/B).
# Appended to ARCH_ARGS so train, validate, and ensemble stay consistent — they MUST
# agree, since the flag changes the forward pass but not the checkpoint state_dict keys.
# Note: set it via this env var, NOT as a CLI passthrough (that reaches train only).
if [[ "${SELF_ATTN_PADDING_MASK:-true}" == "true" ]]; then
    ARCH_ARGS+=(--self_attn_padding_mask)
else
    ARCH_ARGS+=(--no-self_attn_padding_mask)
fi

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
    if ! python -m "$TRAIN_MODULE" \
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
            --fm_solver       "$FM_SOLVER" \
            $( [[ "$USE_WANDB" == "true" ]] && echo "--use_wandb" ) \
            "${EXTRA_TRAIN_ARGS[@]}"; then
        echo "  [FAIL] Training failed — $model"
        FAILED+=("${model}/train")
        continue
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo " VALIDATE  :  $model"
    echo "------------------------------------------------------------"
    if ! python -m "$VALIDATE_MODULE" \
            --model      "$model" \
            --checkpoint "$CKPT_PATH" \
            "${ARCH_ARGS[@]}" \
            --batch_size       "$CF_BATCH" \
            --num_obs          "$NUM_OBS" \
            --num_sample_steps "$FM_STEPS" \
            --num_samples      "$CF_FM_SAMPLES" \
            --solver           "$FM_SOLVER"; then
        echo "  [FAIL] Validation failed — $model"
        FAILED+=("${model}/validate")
        continue
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo " ENSEMBLE  :  $model"
    echo "------------------------------------------------------------"
    if ! python -m "$ENSEMBLE_MODULE" \
            --model      "$model" \
            --checkpoint "$CKPT_PATH" \
            "${ARCH_ARGS[@]}" \
            --batch_size       "$CF_BATCH" \
            --num_obs          "$NUM_OBS" \
            --num_sample_steps "$FM_STEPS" \
            --num_samples      "$CF_FM_SAMPLES" \
            --solver           "$FM_SOLVER"; then
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
