#!/bin/bash
# Design-specific fine-tune of a COPY of the r4c4 model-class checkpoint at the BayesFlow
# baseline's own budget and learning rate: RDM fixed_regressed only (--fixed_design),
# lr 1e-4, 1000 epochs x 100 steps x batch 128 = 12.8M simulated datasets, exactly the
# per-case baseline budget. If CogFormer -- pretrained AND given the baseline's data --
# still falls short of the baseline on this one design, the encoder is the bottleneck.
# Then re-score on the shared bench5 files and C2ST against the existing RDM baselines.
# The original's sha256 is verified before and after; outputs live under the
# family_finetune/rdm_fixedreg_lr1e4_e1000 tag.
set -euo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2
source .venv/bin/activate
export MPLBACKEND=Agg COGFORMER_SIM_BACKEND=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=. NUM_GPU_JOBS=2
log() { echo "$(date '+%F %T') $*"; }

TAG=rdm_fixedreg_lr1e4_e1000
ORIG=./outputs/checkpoints/model_class/cf/r4c4_b128_e15000/cogformer_model_class_fm_l8_h8_p256_s32_d128_b128_e15000_t100.pt
SNAP=./outputs/checkpoints/model_class/cf/family_finetune/rdm_e200/base_snapshot_r4c4_b128_e15000.pt
CKPT_DIR=./outputs/checkpoints/model_class/cf/family_finetune/$TAG
FT_CKPT=$CKPT_DIR/cogformer_model_class_fm_l8_h8_p256_s32_d128_b128_e1000_t100.pt
FIG=./outputs/figures/model_class/cf/family_finetune/$TAG
PRED=./outputs/data/model_class/family_finetune/${TAG}_validate
C2ST_OUT=./outputs/figures/c2st_family_finetune/$TAG/rdm
ARCH=(--self_attn_padding_mask --encoder_num_layers 8 --decoder_num_layers 8
      --encoder_num_heads 8 --decoder_num_heads 8 --proj_dim 256 --num_seeds 32
      --seed_dim 128 --model_embed_dim 8 --max_num_regressors 4 --max_num_categories 4)

mkdir -p "$CKPT_DIR"
[[ -f "$CKPT_DIR/orig.sha256" ]] || sha256sum "$ORIG" > "$CKPT_DIR/orig.sha256"
[[ "$(sha256sum < "$SNAP" | cut -d' ' -f1)" == "$(cut -d' ' -f1 "$CKPT_DIR/orig.sha256")" ]]
log "snapshot identical to the original: $SNAP"

if [[ ! -f "$FT_CKPT.done" ]]; then
    log "train: RDM fixed_regressed-only fine-tune, lr 1e-4, 1000 epochs (baseline budget)"
    # Checkpoint (and a cheap single-scenario validation) every 250 epochs, so a crash
    # loses at most a quarter of the run; --resume picks it up.
    python -m experiments.experiment_1.model_class_cf_train "${ARCH[@]}" \
        --checkpoint_dir "$CKPT_DIR" --init_checkpoint "$SNAP" --family_weights 0,1,0 \
        --fixed_design rdm:fixed_regressed \
        --min_num_obs 500 --max_num_obs 500 --train_batch_size 128 \
        --val_batch_size 50 --val_samples_chunk_size 1 \
        --epochs 1000 --steps_per_epoch 100 --lr 1e-4 --dropout 0.05 --layer_dropout 0.05 \
        --fm_sample_steps 50 --fm_num_samples 100 \
        --val_epoch_interval 250 --val_scenario_tags fixed_regressed --resume \
        --fig_dir "$FIG/train" --use_wandb --run_tag "rdm_fixedreg_lr1e4_e1000"
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
