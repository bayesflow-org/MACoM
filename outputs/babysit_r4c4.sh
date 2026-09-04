#!/usr/bin/env bash
# babysit_r4c4.sh — one-shot orchestration, 2026-07-13:
#   1. wait for ddm r3c2 (pid $DDM_OLD) to exit
#   2. launch rdm r4c4 on the freed GPU 1 (CPUs 24-47)
#   3. wait for the first of {cdm r4c4, rdm r4c4} to exit
#   4. launch ddm r4c4 on whichever GPU that freed
# Detached-safe: no dependence on the launching shell. Progress -> stdout (redirected to log).
set -uo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2
PY=.venv/bin/python

DDM_OLD=3009711   # ddm r3c2 on GPU 1
CDM_PID=91949     # cdm r4c4 on GPU 0

log() { echo "[$(date '+%F %T')] $*"; }

launch_r4c4() { # model gpu cpus -> prints python pid
    local model=$1 gpu=$2 cpus=$3
    local out="outputs/logs/${model}_r4c4_$(date +%Y%m%d_%H%M%S).log"
    taskset -c "$cpus" env CUDA_VISIBLE_DEVICES="$gpu" NUM_GPU_JOBS=2 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        nohup "$PY" -m experiments.model_family.model_family_cf_train \
        --model "$model" --max_num_regressors 4 --max_num_categories 4 \
        --encoder_num_layers 8 --decoder_num_layers 8 \
        --encoder_num_heads 8 --decoder_num_heads 8 \
        --proj_dim 256 --num_seeds 32 --seed_dim 64 \
        --num_obs 500 --min_num_obs 200 --max_num_obs 1000 \
        --train_batch_size 256 --val_batch_size 100 --val_samples_chunk_size 1 \
        --epochs 10000 --steps_per_epoch 100 \
        --lr 1e-4 --dropout 0.05 --layer_dropout 0.05 \
        --fm_sample_steps 25 --fm_num_samples 100 --fm_solver midpoint \
        --val_epoch_interval 2500 --num_sim_workers 20 \
        --checkpoint_dir "./outputs/checkpoints/model_family/cf/${model}/r4c4/" \
        --use_wandb > "$out" 2>&1 &
    echo $!
}

log "waiting for ddm r3c2 (pid $DDM_OLD) to exit"
while kill -0 "$DDM_OLD" 2>/dev/null; do sleep 60; done
log "ddm r3c2 exited"

# Safety net: the grid scheduler is already dead, but kill any straggler
# cf_train bound to GPU 1 before claiming it.
for p in $(pgrep -f 'experiments.model_family.model_family_cf_train'); do
    if tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx 'CUDA_VISIBLE_DEVICES=1'; then
        log "killing straggler GPU1 job pid $p"
        kill -- "-$(ps -o pgid= -p "$p" | tr -d ' ')" 2>/dev/null || kill "$p"
    fi
done
sleep 30

RDM_PID=$(launch_r4c4 rdm 1 24-47)
log "launched rdm r4c4 pid $RDM_PID on GPU1 (cpus 24-47)"

log "speed race: cdm r4c4 pid $CDM_PID (GPU0) vs rdm r4c4 pid $RDM_PID (GPU1)"
while true; do
    if ! kill -0 "$CDM_PID" 2>/dev/null; then FREED_GPU=0; FREED_CPUS=0-23; FIRST_DONE=cdm; break; fi
    if ! kill -0 "$RDM_PID" 2>/dev/null; then FREED_GPU=1; FREED_CPUS=24-47; FIRST_DONE=rdm; break; fi
    sleep 120
done
log "$FIRST_DONE r4c4 exited first; GPU$FREED_GPU is free"
sleep 30

DDM_PID=$(launch_r4c4 ddm "$FREED_GPU" "$FREED_CPUS")
log "launched ddm r4c4 pid $DDM_PID on GPU$FREED_GPU (cpus $FREED_CPUS)"
log "babysitter done"
