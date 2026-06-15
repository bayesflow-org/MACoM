#!/usr/bin/env bash
# run_all.sh — orchestration of the CogFormer experiment suite.
#
# GPU scheduling (when >=2 GPUs): CF training units are paired across the two
# GPUs in waves, one single-GPU job per GPU. Validation runs immediately after
# its own training inside the same job, and we barrier (wait) for BOTH GPUs to
# finish a wave before starting the next, so neither GPU starts new work until
# the wave is fully decommissioned.
#
#   Wave 1:  GPU0 = ModelFamily DDM (train -> validate)
#            GPU1 = ModelFamily CDM (train -> validate)
#   Wave 2:  GPU0 = ModelFamily RDM (train -> validate)
#            GPU1 = ModelClass        (train -> validate -> LBA)
#   Wave 3:  GPU0 = ModelFamily DDM  (BF train+validate -> C2ST)
#            GPU1 = ModelFamily CDM  (BF train+validate -> C2ST)
#   Wave 4:  GPU0 = ModelFamily RDM  (BF train+validate -> C2ST)
#            GPU1 = real_data apply   (needs only CF/MC checkpoints, no BF)
#
# Dependencies honored: C2ST for a model runs in the same job right after that
# model's BF (and its CF outputs exist from waves 1-2); real_data depends only
# on the CF/MC checkpoints produced in waves 1-2, so it pairs freely in wave 4.
#
# Concurrent jobs share one CPU pool, so paired waves export NUM_GPU_JOBS=2:
# each sim worker sizes its numba pool as cpu_count // (n_workers * NUM_GPU_JOBS),
# keeping total threads == cpu_count instead of oversubscribing it 2x. Serial
# phases leave it unset (=1) so they get the full CPU.
#
# Generalization and encoder_rep are intentionally NOT run here.
# Ablations are run separately via experiments/ablations/run_ablations.sh.
#
# Environment variables (forwarded to sub-scripts automatically):
#   DRY_EPOCHS         — dry-run epoch count for CF training
#   MF_BF_DRY_EPOCHS   — dry-run epoch count for BF training
#   USE_WANDB          — set to any non-empty value to enable wandb logging
#
# Usage:
#   bash experiments/run_all.sh 2>&1 | tee outputs/logs/run_all.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p outputs/logs outputs/timing

MF_RUN="${SCRIPT_DIR}/model_family/run.sh"
MC_RUN="${SCRIPT_DIR}/model_class/run.sh"
RD_RUN="${SCRIPT_DIR}/real_data/run.sh"

# Detect total number of GPUs on this machine.
N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)

echo "=== CogFormer experiment suite (N_GPUS=${N_GPUS}) ==="
echo ""

# Stagger the 2nd job of each wave so two identical cold starts don't spike RAM
# at the same instant (each job briefly spawns many torch-importing inductor
# compile workers + data workers). Simultaneous cold starts on this shared box
# got a job OOM-killed; staggering also lets the 1st job populate the shared
# torch.compile cache so the 2nd reuses it.
STAGGER_SECS="${STAGGER_SECS:-90}"
# Cap inductor compile-worker fan-out per job (default ~min(32,ncpu) torch-
# importing subprocesses PER job — the main cold-start RAM bomb when 2 run).
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-12}"

if [ "$N_GPUS" -ge 2 ]; then
    # ── Wave 1: DDM (GPU0) || CDM (GPU1) — CF train+validate only ──────────────
    echo "--- Wave 1: ModelFamily DDM (GPU0) || ModelFamily CDM (GPU1), NUM_GPU_JOBS=2 ---"
    NUM_GPU_JOBS=2 CUDA_VISIBLE_DEVICES=0 CF_MODELS=ddm BF_MODELS= C2ST_MODELS= \
        bash "$MF_RUN" 2>&1 | sed 's/^/[mf-ddm] /' &
    pid_a=$!
    sleep "$STAGGER_SECS"   # stagger 2nd cold start to flatten the RAM spike
    NUM_GPU_JOBS=2 CUDA_VISIBLE_DEVICES=1 CF_MODELS=cdm BF_MODELS= C2ST_MODELS= \
        bash "$MF_RUN" 2>&1 | sed 's/^/[mf-cdm] /' &
    pid_b=$!
    wait $pid_a $pid_b

    # ── Wave 2: RDM (GPU0) || ModelClass (GPU1) ───────────────────────────────
    echo "--- Wave 2: ModelFamily RDM (GPU0) || ModelClass (GPU1), NUM_GPU_JOBS=2 ---"
    NUM_GPU_JOBS=2 CUDA_VISIBLE_DEVICES=0 CF_MODELS=rdm BF_MODELS= C2ST_MODELS= \
        bash "$MF_RUN" 2>&1 | sed 's/^/[mf-rdm] /' &
    pid_a=$!
    sleep "$STAGGER_SECS"   # stagger 2nd cold start to flatten the RAM spike
    NUM_GPU_JOBS=2 CUDA_VISIBLE_DEVICES=1 \
        bash "$MC_RUN" 2>&1 | sed 's/^/[model_class] /' &
    pid_b=$!
    wait $pid_a $pid_b

    # ── Wave 3: BF+C2ST DDM (GPU0) || BF+C2ST CDM (GPU1) ──────────────────────
    echo "--- Wave 3: BF+C2ST DDM (GPU0) || BF+C2ST CDM (GPU1), NUM_GPU_JOBS=2 ---"
    NUM_GPU_JOBS=2 CUDA_VISIBLE_DEVICES=0 CF_MODELS= BF_MODELS=ddm C2ST_MODELS=ddm \
        bash "$MF_RUN" 2>&1 | sed 's/^/[mf-bf-ddm] /' &
    pid_a=$!
    sleep "$STAGGER_SECS"   # stagger 2nd cold start to flatten the RAM spike
    NUM_GPU_JOBS=2 CUDA_VISIBLE_DEVICES=1 CF_MODELS= BF_MODELS=cdm C2ST_MODELS=cdm \
        bash "$MF_RUN" 2>&1 | sed 's/^/[mf-bf-cdm] /' &
    pid_b=$!
    wait $pid_a $pid_b

    # ── Wave 4: BF+C2ST RDM (GPU0) || real_data apply (GPU1) ──────────────────
    echo "--- Wave 4: BF+C2ST RDM (GPU0) || real_data (GPU1), NUM_GPU_JOBS=2 ---"
    NUM_GPU_JOBS=2 CUDA_VISIBLE_DEVICES=0 CF_MODELS= BF_MODELS=rdm C2ST_MODELS=rdm \
        bash "$MF_RUN" 2>&1 | sed 's/^/[mf-bf-rdm] /' &
    pid_a=$!
    sleep "$STAGGER_SECS"   # stagger 2nd cold start to flatten the RAM spike
    NUM_GPU_JOBS=2 CUDA_VISIBLE_DEVICES=1 \
        bash "$RD_RUN" 2>&1 | sed 's/^/[real_data] /' &
    pid_b=$!
    wait $pid_a $pid_b
else
    echo "--- ModelFamily (CF + BF + C2ST) ---"
    bash "$MF_RUN"

    echo "--- ModelClass ---"
    bash "$MC_RUN"

    echo "--- Real data ---"
    bash "$RD_RUN"
fi

echo ""
echo "=== All experiments complete. ==="
python -m experiments.timing_report
