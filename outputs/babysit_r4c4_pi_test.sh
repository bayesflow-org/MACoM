#!/usr/bin/env bash
# babysit_r4c4_pi_test.sh — short r4c4 pi-calibration confirmation runs, 2026-07-20:
#   Confirms the Beta(1,50) contamination-prior change (commit 4c4bac1) fixes pi
#   calibration. Runs ddm then cdm sequentially on GPU0 (cpus 0-23, freed by the
#   completed r4c4 production runs). rdm is skipped — its production r4c4 run is
#   still live on GPU1 and shares the same figure-output path per model name.
#   5000 epochs, batch 128, val_epoch_interval 2500 (2 val steps: 2500/5000),
#   3 val scenarios (intercept_only, fixed_regressed, interaction), wandb on,
#   recovery + ecdf plots. All output under outputs/model_class/val_step/<model>/.
set -uo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2
PY=.venv/bin/python

log() { echo "[$(date '+%F %T')] $*"; }

run_pi_test() { # model -> blocks until exit, logs to outputs/logs/
    local model=$1
    local out="outputs/logs/${model}_r4c4_pi_test_$(date +%Y%m%d_%H%M%S).log"
    local base="outputs/model_class/val_step/${model}"
    log "launching ${model} r4c4 pi-calibration test on GPU0 (cpus 0-23)"
    taskset -c 0-23 env CUDA_VISIBLE_DEVICES=0 NUM_GPU_JOBS=2 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$PY" -m experiments.model_family.model_family_cf_train \
        --model "$model" --max_num_regressors 4 --max_num_categories 4 \
        --encoder_num_layers 8 --decoder_num_layers 8 \
        --encoder_num_heads 8 --decoder_num_heads 8 \
        --proj_dim 256 --num_seeds 32 --seed_dim 64 \
        --num_obs 500 --min_num_obs 200 --max_num_obs 1000 \
        --train_batch_size 128 --val_batch_size 100 --val_samples_chunk_size 1 \
        --epochs 5000 --steps_per_epoch 100 \
        --lr 1e-4 --dropout 0.05 --layer_dropout 0.05 \
        --fm_sample_steps 25 --fm_num_samples 100 --fm_solver midpoint \
        --val_epoch_interval 2500 --num_sim_workers 20 \
        --val_scenario_tags intercept_only,fixed_regressed,interaction \
        --checkpoint_dir "./${base}/checkpoints/" \
        --fig_dir "./${base}/figures/" \
        --use_wandb > "$out" 2>&1
    log "${model} r4c4 pi-calibration test exited (status $?), log: $out"
}

run_pi_test ddm
run_pi_test cdm
log "babysitter done"
