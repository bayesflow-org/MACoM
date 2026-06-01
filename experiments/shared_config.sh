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
BATCH=64
MC_BATCH=64
VAL_BATCH=200

# ── Training ───────────────────────────────────────────────────────────────────
EPOCHS=5000
STEPS=100
MF_BF_EPOCHS=1000
LCA_FFI_EPOCHS=500

# ── Derived checkpoint paths ───────────────────────────────────────────────────
DDM_MF_CKPT="./outputs/checkpoints/model_family/cf/ddm/cogformer_ddm_mf_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${DIM}_o${OBS}_b${BATCH}_e${EPOCHS}_t${STEPS}.pt"
CDM_MF_CKPT="./outputs/checkpoints/model_family/cf/cdm/cogformer_cdm_mf_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${DIM}_o${OBS}_b${BATCH}_e${EPOCHS}_t${STEPS}.pt"
RDM_MF_CKPT="./outputs/checkpoints/model_family/cf/rdm/cogformer_rdm_mf_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${DIM}_o${OBS}_b${BATCH}_e${EPOCHS}_t${STEPS}.pt"

MC_CKPT="./outputs/checkpoints/model_class/cf/cogformer_model_class_fm_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${MC_DIM}_b${MC_BATCH}_e${EPOCHS}_t${STEPS}.pt"

ENC_REP_CKPT="./outputs/checkpoints/encoder_rep/cf/cogformer_encoder_rep_fm_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${MC_DIM}_b${BATCH}_e${EPOCHS}_t${STEPS}.pt"
