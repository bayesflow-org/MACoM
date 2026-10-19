#!/usr/bin/env bash
# Cross-evaluation (Oct 18): each study's ModelClass checkpoint validated on the OTHER
# study's test set, with e1_validate's exact arguments. mixed_predictor is simulated
# on the fly, so it uses the generator of the test set being evaluated on.
#   orig_model_on_gs_data   GPU0   mixture checkpoint  x  gaussian_softmax test set
#   gs_model_on_orig_data   GPU1   gs checkpoint       x  mixture test set
# Each job runs with its own subdirectory as cwd, so no relative path reaches the main tree.
set -u
REPO=/home/jerry/Documents/research/projects/CogFormer-2
GS=$REPO/outputs/gaussian-softmax
HERE=$GS/cross_eval
PY=$REPO/.venv/bin/python
CK=checkpoints/model_class/cf
CKF=cogformer_model_class_fm_l8_h8_p256_s32_d128_b128_e15000_t100.pt
ORIG_CKPT=$REPO/outputs/$CK/r4c4_enc128h4l4_a0.5_b128_e15000_ptc/$CKF
GS_CKPT=$GS/outputs/$CK/r4c4_enc128h4l4_a0.5_b128_e15000_ptc_gaussian_softmax/$CKF
ORIG_DATA=$REPO/outputs/data/benchmark/seed0
GS_DATA=$GS/outputs/data/benchmark/seed0

export PYTHONPATH=$REPO MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       NCCL_P2P_DISABLE=1 COGFORMER_SIM_BACKEND=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
       NUMBA_THREADING_LAYER=workqueue

job() {  # job <name> <gpu> <cpus> <ckpt> <data_dir> <generator>
    local name=$1 gpu=$2 cpus=$3 ckpt=$4 data=$5 gen=$6
    mkdir -p "$HERE/$name"; cd "$HERE/$name"
    echo "[$(date -Is)] start $name (GPU$gpu)" >> "$HERE/status.log"
    CUDA_VISIBLE_DEVICES=$gpu DESIGN_GENERATOR=$gen taskset -c "$cpus" \
    "$PY" -m experiments.experiment_1.model_class_cf_validate \
        --solver rk4 --num_sample_steps 50 --samples_chunk_size 5 --self_attn_padding_mask \
        --encoder_num_layers 4 --decoder_num_layers 8 --encoder_num_heads 4 --decoder_num_heads 8 \
        --encoder_embed_dim 128 --proj_dim 256 --num_seeds 32 --seed_dim 128 --model_embed_dim 8 \
        --max_num_regressors 4 --max_num_categories 4 --design_generator "$gen" \
        --checkpoint "$ckpt" --outdir ./figures --pred_dir ./predictions \
        --batch_size 200 --num_samples 200 --num_obs 500 --seed 0 --skip_posteriors \
        --data_dir "$data" > run.log 2>&1
    echo "[$(date -Is)] end $name rc=$?" >> "$HERE/status.log"
}

job orig_model_on_gs_data 0 0-23  "$ORIG_CKPT" "$GS_DATA"   gaussian_softmax &
job gs_model_on_orig_data 1 24-47 "$GS_CKPT"   "$ORIG_DATA" mixture &
wait
echo "[$(date -Is)] all done" >> "$HERE/status.log"
