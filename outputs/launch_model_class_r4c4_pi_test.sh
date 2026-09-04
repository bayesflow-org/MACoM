#!/usr/bin/env bash
# launch_model_class_r4c4_pi_test.sh — short model_class r4c4 pi-calibration
# confirmation run, 2026-07-20. Confirms the Beta(1,50) contamination-prior
# change (commit 4c4bac1) fixes pi calibration, jointly across DDM/RDM/CDM
# (model_class trains one network for the whole class, unlike model_family).
# 5000 epochs, batch 128, val_epoch_interval 2500 (2 val steps: 2500/5000),
# 3 val scenarios per family (intercept_only, fixed_regressed, interaction),
# wandb on, recovery + ecdf plots. Runs on GPU0 (cpus 0-23, freed by the
# completed r4c4 model_family production runs). All output under
# outputs/model_class/val_step/.
set -uo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2
PY=.venv/bin/python

out="outputs/logs/model_class_r4c4_pi_test_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date '+%F %T')] launching model_class r4c4 pi-calibration test on GPU0 (cpus 0-23)"

taskset -c 0-23 env CUDA_VISIBLE_DEVICES=0 NUM_GPU_JOBS=2 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -m experiments.model_class.model_class_cf_train \
    --max_num_regressors 4 --max_num_categories 4 \
    --encoder_num_layers 8 --decoder_num_layers 8 \
    --encoder_num_heads 8 --decoder_num_heads 8 \
    --proj_dim 256 --num_seeds 32 --seed_dim 128 --model_embed_dim 8 \
    --time_embedding_dim 32 --pos_embedding_dim 32 \
    --min_num_obs 200 --max_num_obs 1000 --val_num_obs 500 \
    --train_batch_size 128 --val_batch_size 100 --val_samples_chunk_size 1 \
    --epochs 5000 --steps_per_epoch 100 \
    --lr 1e-4 --dropout 0.05 --layer_dropout 0.05 \
    --fm_sample_steps 25 --fm_num_samples 100 --fm_solver midpoint \
    --val_epoch_interval 2500 \
    --num_sim_workers 20 \
    --val_scenario_tags intercept_only,fixed_regressed,interaction \
    --checkpoint_dir "./outputs/model_class/val_step/checkpoints/" \
    --fig_dir "./outputs/model_class/val_step/figures/" \
    --use_wandb > "$out" 2>&1

echo "[$(date '+%F %T')] model_class r4c4 pi-calibration test exited (status $?), log: $out"
