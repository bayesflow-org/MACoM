#!/usr/bin/env bash
# run_resume.sh — recovery orchestrator after DDM's co-tenant OOM in the first run.
#
# CDM (original Wave 1) is left RUNNING on GPU1 (we only killed the run_all layer,
# not CDM). This script reruns DDM and pairs it with RDM, then finishes the suite,
# keeping BOTH GPUs busy throughout and honoring data deps:
#
#   now:        DDM (GPU0, fresh)              || CDM (GPU1, finishing Wave 1)
#   CDM done -> DDM (GPU0)                      || RDM (GPU1, fresh)   <-- DDM ∥ RDM
#   DDM done -> ModelClass (GPU0)              || RDM (GPU1)
#   RDM done -> ModelClass (GPU0)              || BF+C2ST ddm (GPU1)
#   MC done  -> BF+C2ST cdm (GPU0)            || BF+C2ST ddm (GPU1)
#   ...      -> BF+C2ST cdm (GPU0)            || BF+C2ST rdm (GPU1)
#   then:       real_data apply
#
# Deps honored: C2ST <model> runs in the same job after that model's BF, and its
# CF outputs already exist; real_data runs last (needs all CF + MC checkpoints).
#
# Not -e: a single sub-job failure should not abort the whole recovery.
set -uo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2

MF=experiments/model_family/run.sh
MC=experiments/model_class/run.sh
RD=experiments/real_data/run.sh
export TORCHINDUCTOR_COMPILE_THREADS=12   # cap compile-worker fan-out (RAM)

gone() { while pgrep -f "$1" >/dev/null 2>&1; do sleep 60; done; }   # block until pattern has no live procs

echo "[resume] $(date '+%F %T') start — DDM on GPU0 now; RDM joins when CDM frees GPU1."

# ── DDM (GPU0): fresh rerun with the val-mem + expandable_segments fixes ─────────
NUM_GPU_JOBS=2 CUDA_VISIBLE_DEVICES=0 CF_MODELS=ddm BF_MODELS= C2ST_MODELS= USE_WANDB=1 \
    bash "$MF" 2>&1 | sed 's/^/[mf-ddm] /' &
ddm=$!

# ── wait for the orphaned CDM (train + validate) to finish and free GPU1 ─────────
echo "[resume] waiting for CDM (Wave 1) to finish ..."
gone "model_family.*--model cdm"
echo "[resume] $(date '+%F %T') CDM done -> launching RDM on GPU1 (DDM ∥ RDM)."

# ── RDM (GPU1): fresh, paired with the still-running DDM ─────────────────────────
NUM_GPU_JOBS=2 CUDA_VISIBLE_DEVICES=1 CF_MODELS=rdm BF_MODELS= C2ST_MODELS= USE_WANDB=1 \
    bash "$MF" 2>&1 | sed 's/^/[mf-rdm] /' &
rdm=$!

# ── ModelClass on GPU0 once DDM finishes ────────────────────────────────────────
wait "$ddm"
echo "[resume] $(date '+%F %T') DDM done -> launching ModelClass on GPU0."
NUM_GPU_JOBS=2 CUDA_VISIBLE_DEVICES=0 USE_WANDB=1 \
    bash "$MC" 2>&1 | sed 's/^/[model_class] /' &
mc=$!

# ── BF+C2ST ddm on GPU1 once RDM finishes ───────────────────────────────────────
wait "$rdm"
echo "[resume] $(date '+%F %T') RDM done -> BF+C2ST ddm on GPU1."
NUM_GPU_JOBS=2 CUDA_VISIBLE_DEVICES=1 CF_MODELS= BF_MODELS=ddm C2ST_MODELS=ddm \
    bash "$MF" 2>&1 | sed 's/^/[mf-bf-ddm] /' &
bfddm=$!

# ── BF+C2ST cdm on GPU0 once ModelClass finishes ────────────────────────────────
wait "$mc"
echo "[resume] $(date '+%F %T') ModelClass done -> BF+C2ST cdm on GPU0."
NUM_GPU_JOBS=2 CUDA_VISIBLE_DEVICES=0 CF_MODELS= BF_MODELS=cdm C2ST_MODELS=cdm \
    bash "$MF" 2>&1 | sed 's/^/[mf-bf-cdm] /' &
bfcdm=$!

# ── BF+C2ST rdm on GPU1 once BF ddm finishes ────────────────────────────────────
wait "$bfddm"
echo "[resume] $(date '+%F %T') BF ddm done -> BF+C2ST rdm on GPU1."
NUM_GPU_JOBS=2 CUDA_VISIBLE_DEVICES=1 CF_MODELS= BF_MODELS=rdm C2ST_MODELS=rdm \
    bash "$MF" 2>&1 | sed 's/^/[mf-bf-rdm] /' &
bfrdm=$!

wait "$bfcdm" "$bfrdm"
echo "[resume] $(date '+%F %T') all BF+C2ST done -> real_data."
bash "$RD" 2>&1 | sed 's/^/[real_data] /'

echo "[resume] $(date '+%F %T') COMPLETE."
