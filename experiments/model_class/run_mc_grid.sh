#!/usr/bin/env bash
# run_mc_grid.sh — sequential ModelClass (DDM+CDM+RDM) CF training over an rXcY
# design grid on one GPU. The model_class analogue of model_family/run_cf_grid.sh.
#
# Each job trains the whole model class at one (max_num_regressors, max_num_categories).
# Checkpoints go to a per-design subdirectory because the checkpoint filename does not
# encode the design space. Unlike the model_family grid there is no <model> field:
# model_class always trains DDM+CDM+RDM jointly, so a job is just "<r>:<c>".
#
# Usage:
#   run_mc_grid.sh "<r>:<c> [<r>:<c> ...]"
#
#   setsid nohup taskset -c 0-23 env CUDA_VISIBLE_DEVICES=0 \
#     bash experiments/model_class/run_mc_grid.sh "2:2 3:3 4:4" \
#     > outputs/logs/mc_cf_grid_gpu0.log 2>&1 &

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Two concurrent single-GPU jobs share the box: divides numba threads so each
# sim worker gets cpu_count / (num_sim_workers * NUM_GPU_JOBS) = 1 thread.
export NUM_GPU_JOBS=2

EPOCHS="${EPOCHS:-5000}"
STEPS="${STEPS:-100}"
BATCH="${BATCH:-64}"
# val_batch_size is capped low on purpose: the model_class default (500) spikes GPU
# memory (see run_resume2.sh notes on the ~41 GB val footprint). 100 + chunk 1 is safe.
VAL_BATCH="${VAL_BATCH:-100}"
VAL_INTERVAL="${VAL_INTERVAL:-2500}"
SIM_WORKERS="${SIM_WORKERS:-20}"

for job in $1; do
    IFS=: read -r r c <<< "$job"
    tag="r${r}c${c}"
    echo "=== [$(date '+%F %T')] model_class ${tag}: train (b${BATCH} e${EPOCHS} t${STEPS}) ==="
    python -m experiments.model_class.model_class_cf_train \
        --max_num_regressors "$r" \
        --max_num_categories "$c" \
        --encoder_num_layers 8 \
        --decoder_num_layers 8 \
        --encoder_num_heads 8 \
        --decoder_num_heads 8 \
        --proj_dim 256 \
        --num_seeds 32 \
        --seed_dim 128 \
        --model_embed_dim 8 \
        --time_embedding_dim 32 \
        --pos_embedding_dim 32 \
        --min_num_obs 200 \
        --max_num_obs 1000 \
        --val_num_obs 500 \
        --train_batch_size "$BATCH" \
        --val_batch_size "$VAL_BATCH" \
        --val_samples_chunk_size 1 \
        --epochs "$EPOCHS" \
        --steps_per_epoch "$STEPS" \
        --lr 1e-4 --dropout 0.05 --layer_dropout 0.05 \
        --fm_sample_steps 25 --fm_num_samples 100 --fm_solver midpoint \
        --val_epoch_interval "$VAL_INTERVAL" \
        --num_sim_workers "$SIM_WORKERS" \
        --checkpoint_dir "./outputs/checkpoints/model_class/cf/${tag}/" \
        --use_wandb
    echo "=== [$(date '+%F %T')] model_class ${tag}: done ==="
done
echo "=== [$(date '+%F %T')] all grid jobs finished ==="
