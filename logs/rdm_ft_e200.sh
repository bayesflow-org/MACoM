#!/bin/bash
# RDM-only fine-tune of a COPY of the r4c4 model-class checkpoint (200 epochs), then
# re-score on the shared bench5 files and C2ST against the existing RDM BF baselines.
# The original checkpoint is read once (to snapshot it) and its sha256 is verified
# before and after. Every output lives under a family_finetune/rdm_e200 tag.
set -euo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2
source .venv/bin/activate
export MPLBACKEND=Agg COGFORMER_SIM_BACKEND=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=. NUM_GPU_JOBS=2
log() { echo "$(date '+%F %T') $*"; }

TAG=rdm_e200
ORIG=./outputs/checkpoints/model_class/cf/r4c4_b128_e15000/cogformer_model_class_fm_l8_h8_p256_s32_d128_b128_e15000_t100.pt
CKPT_DIR=./outputs/checkpoints/model_class/cf/family_finetune/$TAG
SNAP=$CKPT_DIR/base_snapshot_r4c4_b128_e15000.pt
FT_CKPT=$CKPT_DIR/cogformer_model_class_fm_l8_h8_p256_s32_d128_b128_e200_t100.pt
FIG=./outputs/figures/model_class/cf/family_finetune/$TAG
PRED=./outputs/data/model_class/family_finetune/${TAG}_validate
C2ST_OUT=./outputs/figures/c2st_family_finetune/$TAG/rdm
ARCH=(--self_attn_padding_mask --encoder_num_layers 8 --decoder_num_layers 8
      --encoder_num_heads 8 --decoder_num_heads 8 --proj_dim 256 --num_seeds 32
      --seed_dim 128 --model_embed_dim 8 --max_num_regressors 4 --max_num_categories 4)

mkdir -p "$CKPT_DIR"
if [[ ! -f "$CKPT_DIR/orig.sha256" ]]; then
    sha256sum "$ORIG" > "$CKPT_DIR/orig.sha256"
    cp --preserve=timestamps "$ORIG" "$SNAP"
    [[ "$(sha256sum < "$SNAP" | cut -d' ' -f1)" == "$(cut -d' ' -f1 "$CKPT_DIR/orig.sha256")" ]]
    log "snapshot ok: $SNAP"
fi

if [[ ! -f "$FT_CKPT.done" ]]; then
    log "train: RDM-only fine-tune, 200 epochs"
    python -m experiments.experiment_1.model_class_cf_train "${ARCH[@]}" \
        --checkpoint_dir "$CKPT_DIR" --init_checkpoint "$SNAP" --family_weights 0,1,0 \
        --min_num_obs 200 --max_num_obs 1000 --train_batch_size 128 \
        --val_batch_size 50 --val_samples_chunk_size 1 \
        --epochs 200 --steps_per_epoch 100 --lr 1e-5 --dropout 0.05 --layer_dropout 0.05 \
        --fm_sample_steps 50 --fm_num_samples 100 \
        --val_epoch_interval 200 --val_scenario_tags interaction --resume \
        --fig_dir "$FIG/train" --use_wandb --run_tag "rdm_ft_e200"
    touch "$FT_CKPT.done"
fi

log "validate: bench5, 6 scenarios x 3 families"
python -m experiments.experiment_1.model_class_cf_validate "${ARCH[@]}" \
    --checkpoint "$FT_CKPT" --outdir "$FIG/validate" --pred_dir "$PRED" \
    --batch_size 200 --num_samples 200 --num_obs 500 --samples_chunk_size 1 --seed 0 \
    --skip_posteriors --data_dir ./outputs/data/benchmark/seed0

log "C2ST: RDM fine-tuned vs BF baselines"
python -m experiments.experiment_1a.model_family_c2st --model rdm \
    --bf_data_dir ./outputs/data/model_family --mc_pred_dir "$PRED" \
    --scenarios BF_vs_Original --max_num_categories 2 --outdir "$C2ST_OUT"

sha256sum -c "$CKPT_DIR/orig.sha256" && log "original checkpoint unchanged"
log "ALL DONE"
