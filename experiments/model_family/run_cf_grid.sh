#!/usr/bin/env bash
# run_cf_grid.sh — sequential CF training over an rXcY design grid on one GPU.
#
# Usage:
#   run_cf_grid.sh "<model>:<r>:<c> [<model>:<r>:<c> ...]"
#
# Each job trains model_family_cf_train with max_num_regressors=r and
# max_num_categories=c. Checkpoints go to a per-design subdirectory because the
# checkpoint filename does not encode the design space. Intended to be launched
# once per GPU with CUDA_VISIBLE_DEVICES and a taskset CPU partition, e.g.:
#
#   setsid nohup taskset -c 0-23 env CUDA_VISIBLE_DEVICES=0 \
#     bash experiments/model_family/run_cf_grid.sh "ddm:2:2 cdm:3:3" \
#     > outputs/logs/mf_cf_grid_gpu0.log 2>&1 &

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Two concurrent single-GPU jobs share the box: divides numba threads so each
# sim worker gets cpu_count / (num_sim_workers * NUM_GPU_JOBS) = 1 thread.
export NUM_GPU_JOBS=2

EPOCHS="${EPOCHS:-5000}"
STEPS="${STEPS:-100}"
BATCH="${BATCH:-128}"
VAL_BATCH="${VAL_BATCH:-100}"
VAL_INTERVAL="${VAL_INTERVAL:-2500}"
SIM_WORKERS="${SIM_WORKERS:-20}"

for job in $1; do
    IFS=: read -r model r c <<< "$job"
    tag="r${r}c${c}"
    echo "=== [$(date '+%F %T')] ${model} ${tag}: train (b${BATCH} e${EPOCHS} t${STEPS}) ==="
    python -m experiments.model_family.model_family_cf_train \
        --model "$model" \
        --max_num_regressors "$r" \
        --max_num_categories "$c" \
        --encoder_num_layers 8 \
        --decoder_num_layers 8 \
        --encoder_num_heads 8 \
        --decoder_num_heads 8 \
        --proj_dim 256 \
        --num_seeds 32 \
        --seed_dim 64 \
        --num_obs 500 \
        --min_num_obs 200 \
        --max_num_obs 1000 \
        --train_batch_size "$BATCH" \
        --val_batch_size "$VAL_BATCH" \
        --val_samples_chunk_size 1 \
        --epochs "$EPOCHS" \
        --steps_per_epoch "$STEPS" \
        --lr 1e-4 --dropout 0.05 --layer_dropout 0.05 \
        --fm_sample_steps 25 --fm_num_samples 100 --fm_solver midpoint \
        --val_epoch_interval "$VAL_INTERVAL" \
        --num_sim_workers "$SIM_WORKERS" \
        --checkpoint_dir "./outputs/checkpoints/model_family/cf/${model}/${tag}/" \
        --use_wandb
    echo "=== [$(date '+%F %T')] ${model} ${tag}: done ==="
done
echo "=== [$(date '+%F %T')] all grid jobs finished ==="
