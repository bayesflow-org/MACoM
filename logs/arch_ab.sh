#!/bin/bash
# Architecture A/B, one arm: train the model-class CogFormer FROM SCRATCH at a reduced,
# matched budget (2000 epochs; otherwise e1_train's exact recipe), then re-score on the
# shared bench5 files and run the per-family C2ST against the existing BayesFlow
# baselines (no baseline retraining).
# Usage: logs/arch_ab.sh <encoder_embed_dim> [encoder_num_heads=8] [encoder_num_layers=8]
#   64          = the current architecture (control): width 64, 8 heads of 8, 8 blocks
#   128 4 4     = encoder matched to the BayesFlow baseline's SetTransformer summary
#                 network: width 128, 4 heads of 32, 4 blocks (decoder unchanged)
# The trainer takes no seed, so the arms also differ in their random streams.
set -euo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2
source .venv/bin/activate
export MPLBACKEND=Agg COGFORMER_SIM_BACKEND=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=. NUM_GPU_JOBS=2
log() { echo "$(date '+%F %T') $*"; }

EMBED=${1:?usage: arch_ab.sh <encoder_embed_dim> [encoder_num_heads] [encoder_num_layers]}
ENC_HEADS=${2:-8}
ENC_LAYERS=${3:-8}
EPOCHS=2000
TAG=embed${EMBED}_e${EPOCHS}
[[ "$ENC_HEADS" == 8 && "$ENC_LAYERS" == 8 ]] || TAG=embed${EMBED}_h${ENC_HEADS}_l${ENC_LAYERS}_e${EPOCHS}
CKPT_DIR=./outputs/checkpoints/model_class/cf/arch_ab/$TAG
CKPT=$CKPT_DIR/cogformer_model_class_fm_l8_h8_p256_s32_d128_b128_e${EPOCHS}_t100.pt
FIG=./outputs/figures/model_class/cf/arch_ab/$TAG
PRED=./outputs/data/model_class/arch_ab/${TAG}_validate
C2ST_OUT=./outputs/figures/c2st_arch_ab/$TAG
ARCH=(--self_attn_padding_mask --encoder_num_layers "$ENC_LAYERS" --decoder_num_layers 8
      --encoder_num_heads "$ENC_HEADS" --decoder_num_heads 8 --proj_dim 256 --num_seeds 32
      --seed_dim 128 --model_embed_dim 8 --max_num_regressors 4 --max_num_categories 4
      --encoder_embed_dim "$EMBED")

mkdir -p "$CKPT_DIR"
if [[ ! -f "$CKPT.done" ]]; then
    log "train: from scratch, encoder width $EMBED, $ENC_HEADS heads, $ENC_LAYERS blocks, $EPOCHS epochs"
    python -m experiments.experiment_1.model_class_cf_train "${ARCH[@]}" \
        --checkpoint_dir "$CKPT_DIR" \
        --min_num_obs 200 --max_num_obs 1000 --train_batch_size 128 \
        --val_batch_size 50 --val_samples_chunk_size 1 \
        --epochs "$EPOCHS" --steps_per_epoch 100 --lr 1e-4 --dropout 0.05 --layer_dropout 0.05 \
        --fm_sample_steps 50 --fm_num_samples 100 \
        --val_epoch_interval 500 --resume \
        --fig_dir "$FIG/train" --use_wandb --run_tag "arch_ab_$TAG"
    touch "$CKPT.done"
fi

log "validate: bench5, 6 scenarios x 3 families"
python -m experiments.experiment_1.model_class_cf_validate "${ARCH[@]}" \
    --checkpoint "$CKPT" --outdir "$FIG/validate" --pred_dir "$PRED" \
    --batch_size 200 --num_samples 200 --num_obs 500 --samples_chunk_size 1 --seed 0 \
    --skip_posteriors --data_dir ./outputs/data/benchmark/seed0

for fam in ddm rdm cdm; do
    log "C2ST: $fam vs BF baselines"
    python -m experiments.experiment_1a.model_family_c2st --model "$fam" \
        --bf_data_dir ./outputs/data/model_family --mc_pred_dir "$PRED" \
        --scenarios BF_vs_Original --max_num_categories 2 --outdir "$C2ST_OUT/$fam"
done
log "ALL DONE"
