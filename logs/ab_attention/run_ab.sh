#!/usr/bin/env bash
# 2026-09-27 A/B: BayesFlow-style attention options (gated_residual + pre_pool_ffn +
# orthogonal init) vs current CogFormer. Final-run recipe (r4c4, b128) at 500 epochs,
# full cosine schedule. Both arms validated on identical seeded datasets.
cd /home/a_huangm13/Documents/Research/Projects/CogFormer-2
source experiments/shared_config.sh
export PYTHONPATH=$PWD MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1
L=logs/ab_attention; R=outputs/ab_attention; EP=500
ARCH=(--self_attn_padding_mask --encoder_num_layers $ENC_LAYERS --decoder_num_layers $DEC_LAYERS
      --encoder_num_heads $HEADS --decoder_num_heads $HEADS --proj_dim $PROJ --num_seeds $SEEDS
      --seed_dim $MC_DIM --model_embed_dim 8 --max_num_regressors $MC_MAX_REG --max_num_categories $MC_MAX_CAT)
declare -A FLAGS=([off]="" [on]="--gated_residual --pre_pool_ffn")
log() { echo "[$(date '+%F %T')] $*" >> $L/ab.log; }
for arm in off on; do
    extra=${FLAGS[$arm]}; init=default; [[ $arm == on ]] && init=orthogonal
    log "train $arm start"
    python -m experiments.experiment_1.model_class_cf_train "${ARCH[@]}" $extra --weight_init $init \
        --checkpoint_dir $R/$arm/ckpt --fig_dir $R/$arm/train_figs \
        --min_num_obs $MIN_OBS --max_num_obs $MAX_OBS --train_batch_size 128 \
        --epochs $EP --steps_per_epoch $STEPS --val_epoch_interval 100000 \
        --lr 1e-4 --dropout 0.05 --layer_dropout 0.05 > $L/train_$arm.log 2>&1 \
        || { log "train $arm FAILED"; echo fail > $L/ab.done; exit 1; }
    log "train $arm done"
done
for arm in off on; do
    extra=${FLAGS[$arm]}
    log "validate $arm start"
    python -m experiments.experiment_1.model_class_cf_validate "${ARCH[@]}" $extra \
        --checkpoint $(ls $R/$arm/ckpt/*.pt) --outdir $R/$arm/validate --pred_dir $R/$arm/pred \
        --batch_size 200 --num_samples 200 --num_obs $OBS --samples_chunk_size 1 --seed 0 \
        > $L/validate_$arm.log 2>&1 || { log "validate $arm FAILED"; echo fail > $L/ab.done; exit 1; }
    log "validate $arm done"
done
echo ok > $L/ab.done
