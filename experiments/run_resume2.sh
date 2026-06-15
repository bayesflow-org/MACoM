#!/usr/bin/env bash
# run_resume2.sh — second recovery pass.
#
# Context: original DDM and CDM both OOM'd on their validation step (old
# val_batch_size=200, ~41 GB spike). The val footprint is now capped in
# model_family/run.sh (val_batch_size 50, chunk 1) + expandable_segments.
# DDM (rerun) and RDM are currently training healthily with the fix and will
# finish on their own (their run.sh wrappers were orphaned, not killed).
#
# CDM is an incomplete hole (died at epoch 2500). CDM and ModelClass are the
# remaining unpaired CF units, so we pair them:
#
#   wait for DDM + RDM (in-flight) to finish
#   -> CDM (GPU0, fresh+fixed)        || ModelClass (GPU1)
#   -> BF+C2ST ddm (GPU0)             || ModelClass (GPU1)
#   -> BF+C2ST ddm (GPU0)            ... then BF+C2ST cdm (GPU1)
#   -> BF+C2ST rdm (GPU0)            || BF+C2ST cdm (GPU1)
#   -> real_data
#
# Deps: C2ST <model> runs after that model's BF (same job) and its CF outputs;
# real_data runs last (needs all CF + MC checkpoints).
set -uo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2

MF=experiments/model_family/run.sh
MC=experiments/model_class/run.sh
RD=experiments/real_data/run.sh
export TORCHINDUCTOR_COMPILE_THREADS=12
gone() { while pgrep -f "$1" >/dev/null 2>&1; do sleep 60; done; }

echo "[resume2] $(date '+%F %T') waiting for in-flight DDM + RDM to finish ..."
gone "model_family.*--model ddm"
gone "model_family.*--model rdm"
echo "[resume2] $(date '+%F %T') DDM+RDM done -> CDM (GPU0) || ModelClass (GPU1)"

# CDM rerun (GPU0) with val fix, paired with ModelClass (GPU1)
NUM_GPU_JOBS=2 CUDA_VISIBLE_DEVICES=0 CF_MODELS=cdm BF_MODELS= C2ST_MODELS= USE_WANDB=1 \
    bash "$MF" 2>&1 | sed 's/^/[mf-cdm2] /' &
cdm=$!
sleep 90
NUM_GPU_JOBS=2 CUDA_VISIBLE_DEVICES=1 USE_WANDB=1 \
    bash "$MC" 2>&1 | sed 's/^/[model_class] /' &
mc=$!

wait "$cdm"
echo "[resume2] $(date '+%F %T') CDM done -> BF+C2ST ddm (GPU0)"
NUM_GPU_JOBS=2 CUDA_VISIBLE_DEVICES=0 CF_MODELS= BF_MODELS=ddm C2ST_MODELS=ddm \
    bash "$MF" 2>&1 | sed 's/^/[mf-bf-ddm] /' &
bfddm=$!

wait "$mc"
echo "[resume2] $(date '+%F %T') ModelClass done -> BF+C2ST cdm (GPU1)"
NUM_GPU_JOBS=2 CUDA_VISIBLE_DEVICES=1 CF_MODELS= BF_MODELS=cdm C2ST_MODELS=cdm \
    bash "$MF" 2>&1 | sed 's/^/[mf-bf-cdm] /' &
bfcdm=$!

wait "$bfddm"
echo "[resume2] $(date '+%F %T') BF ddm done -> BF+C2ST rdm (GPU0)"
NUM_GPU_JOBS=2 CUDA_VISIBLE_DEVICES=0 CF_MODELS= BF_MODELS=rdm C2ST_MODELS=rdm \
    bash "$MF" 2>&1 | sed 's/^/[mf-bf-rdm] /' &
bfrdm=$!

wait "$bfcdm" "$bfrdm"
echo "[resume2] $(date '+%F %T') all BF+C2ST done -> real_data"
bash "$RD" 2>&1 | sed 's/^/[real_data] /'
echo "[resume2] $(date '+%F %T') COMPLETE"
