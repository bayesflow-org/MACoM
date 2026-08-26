#!/usr/bin/env bash
# shared_config.sh — hyperparameters and derived checkpoint paths for all
# CogFormer experiments. Source this file; do not execute it directly.
#
# Usage (from any experiment script):
#   source "$(dirname "${BASH_SOURCE[0]}")/../shared_config.sh"   # from a sub-dir
#   source "$(dirname "${BASH_SOURCE[0]}")/shared_config.sh"      # from experiments/

# ── Architecture ───────────────────────────────────────────────────────────────
ENC_LAYERS=8
DEC_LAYERS=8
HEADS=8
PROJ=256
SEEDS=32
DIM=64       # seed_dim for model-family / generalization runs
MC_DIM=128   # seed_dim for model-class / encoder-rep runs

# ── Data ───────────────────────────────────────────────────────────────────────
OBS=500
MIN_OBS=200
MAX_OBS=1000

# Batch size — set to the measured throughput peak, not the VRAM max.
# CogFormer's O(N²) decoder self-attention saturates the GPU even at small batch,
# so samples/sec PEAKS at batch=64 and DECLINES as batch grows (bench_batch.py on
# RTX 6000 Ada, mask ON: 64→507, 128→468, 256→438 samples/s). batch=64 is also
# the most simulation-budget-per-hour and uses ~8GB VRAM (huge OOM headroom).
# To grow the simulation budget, raise EPOCHS/STEPS — not the batch.
if [ -z "${BATCH:-}" ]; then BATCH=64; MC_BATCH=64; fi

VAL_BATCH=200

# ── Training ───────────────────────────────────────────────────────────────────
# EPOCHS=10000 at batch=64/steps=100 → 64M simulated datasets/model (matches the
# old batch=128 × 5000ep budget) at slightly less wall-clock and 2× gradient updates.
EPOCHS=10000
STEPS=100
MF_BF_EPOCHS=1000
LCA_FFI_EPOCHS=500
LBA_FT_EPOCHS=2000

# ── Encoder design geometry ────────────────────────────────────────────────────
# A checkpoint's encoder input_embedding is sized by (max_num_regressors,
# max_num_categories) via build_encoder_input_dim: r2c2→9, r2c3→14, r4c4→72,
# r5c5→186. A checkpoint can ONLY be loaded at the geometry it was trained at,
# so these travel with the paths below and must be passed to every apply script.
# The real-data designs need: DDM/RDM 2 continuous regressors → r2c2;
# CDM 2 regressors with a 3-level categorical (difficulty) → r2c3.
MF_MAX_REG=2; MF_MAX_CAT=2      # ddm, rdm
CDM_MF_MAX_REG=2; CDM_MF_MAX_CAT=3
MC_MAX_REG=4; MC_MAX_CAT=4

# ── Derived checkpoint paths ───────────────────────────────────────────────────
# NOTE: every checkpoint predating commit 77be0b0 (2026-07-09, factorized
# regressor-category identity embedding) lacks decoder.{regressor,category}_embedding
# and CANNOT be loaded by current code. That rules out everything under */c2r2/.
DDM_MF_CKPT="./outputs/checkpoints/model_family/cf/ddm/r2c2/cogformer_ddm_mf_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${DIM}_o${OBS}_b128_e2500_t${STEPS}.pt"
# MISSING — no post-factorization CDM/RDM model-family checkpoint exists yet.
# Newest are ./outputs/checkpoints/model_family/cf/{cdm,rdm}/c2r2/... (2026-07-08),
# which are pre-factorization. Both need retraining before real-data apply can run.
CDM_MF_CKPT="./outputs/checkpoints/model_family/cf/cdm/r2c3/cogformer_cdm_mf_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${DIM}_o${OBS}_b128_e2500_t${STEPS}.pt"
RDM_MF_CKPT="./outputs/checkpoints/model_family/cf/rdm/r2c2/cogformer_rdm_mf_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${DIM}_o${OBS}_b128_e2500_t${STEPS}.pt"

MC_CKPT="./outputs/checkpoints/model_class/cf/r4c4_b256_e10000/cogformer_model_class_fm_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${MC_DIM}_b256_e${EPOCHS}_t${STEPS}.pt"

ENC_REP_CKPT="./outputs/checkpoints/encoder_rep/cf/cogformer_encoder_rep_fm_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${MC_DIM}_b${BATCH}_e${EPOCHS}_t${STEPS}.pt"

# LBA_FIXED_A is the value every LBA arm must use -- finetune, zero-shot and
# validation alike. It is in the checkpoint filename because fixed_A changes what
# the intrinsic `a` MEANS (threshold vs start-point range), so runs at different
# values are not comparable. The batch size is literal 256, not MC_BATCH: the
# finetune ran at 256 while MC_BATCH is 64, and the previous interpolation
# produced a path that did not exist on disk.
LBA_FIXED_A=0.5
LBA_FT_BATCH=256
LBA_FT_CKPT="./outputs/checkpoints/model_class/cf/lba_finetune/cogformer_lba_finetune_s2_A${LBA_FIXED_A}_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${MC_DIM}_b${LBA_FT_BATCH}_e${LBA_FT_EPOCHS}_t${STEPS}.pt"
