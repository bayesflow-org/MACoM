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

# ── Derived checkpoint paths ───────────────────────────────────────────────────
DDM_MF_CKPT="./outputs/checkpoints/model_family/cf/ddm/cogformer_ddm_mf_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${DIM}_o${OBS}_b${BATCH}_e${EPOCHS}_t${STEPS}.pt"
CDM_MF_CKPT="./outputs/checkpoints/model_family/cf/cdm/cogformer_cdm_mf_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${DIM}_o${OBS}_b${BATCH}_e${EPOCHS}_t${STEPS}.pt"
RDM_MF_CKPT="./outputs/checkpoints/model_family/cf/rdm/cogformer_rdm_mf_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${DIM}_o${OBS}_b${BATCH}_e${EPOCHS}_t${STEPS}.pt"

MC_CKPT="./outputs/checkpoints/model_class/cf/cogformer_model_class_fm_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${MC_DIM}_b${MC_BATCH}_e${EPOCHS}_t${STEPS}.pt"

ENC_REP_CKPT="./outputs/checkpoints/encoder_rep/cf/cogformer_encoder_rep_fm_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${MC_DIM}_b${BATCH}_e${EPOCHS}_t${STEPS}.pt"

LBA_FT_CKPT="./outputs/checkpoints/model_class/cf/lba_finetune/cogformer_lba_finetune_s2_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${MC_DIM}_b${MC_BATCH}_e${LBA_FT_EPOCHS}_t${STEPS}.pt"
