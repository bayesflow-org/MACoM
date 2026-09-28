#!/usr/bin/env bash
# shared_config.sh — hyperparameters and derived checkpoint paths for the finals
# pipeline (run_experiments.sh). Source this file; do not execute it directly.
# experiments/shared_config.py reads the same values for the Python assemblers.

# ── Architecture ───────────────────────────────────────────────────────────────
ENC_LAYERS=8
DEC_LAYERS=8
HEADS=8
PROJ=256
SEEDS=32
MC_DIM=128   # seed_dim of the model-class checkpoint

# ── Data ───────────────────────────────────────────────────────────────────────
OBS=500
MIN_OBS=200
MAX_OBS=1000
STEPS=100

# ── Final-run budget (decided 2026-09-27) ──────────────────────────────────────
# Equal simulation budget AND equal batch size across the amortization-gap arms:
#   ModelClass CogFormer : 15000 ep x 100 steps x 128 = 192M datasets (3 families)
#   BayesFlow baselines  :  1000 ep x 100 steps x 128 x 15 cases = 192M datasets
# i.e. 64M per family on both sides -- the predecessor's per-family equivalence,
# lifted to the whole class. Measured 14 s/epoch at b128, so ~58 h. SIMULATION-bound
# on a 24-core host: 20 workers deliver ~7.5 batches/s vs ~9.7 steps/s the GPU can take
# (2026-09-28 profile), so a faster GPU step alone does not shorten e1_train here.
MC_EPOCHS=15000
MC_TRAIN_BATCH=128
MF_BF_EPOCHS=1000
MF_BF_BATCH=128
# 1000, not 2000: finetuning gains plateau by ~1000 epochs (2026-09-27).
LBA_FT_EPOCHS=1000

# Simulation backend for TRAINING stages (e1_train, e3_finetune, e1a_bf). "gpu" runs
# DDM/RDM/CDM on Triton kernels (LBA/aDDM stay on numba): distributionally equivalent
# to numba (experiments/check_gpu_simulators.py) and it lifts the simulation ceiling
# that bound training on CPU. Data generation and validation stay on numba, so every
# seeded/paired test set is produced exactly as before.
SIM_BACKEND=gpu

# ── Encoder design geometry ────────────────────────────────────────────────────
# A checkpoint's encoder input_embedding is sized by (max_num_regressors,
# max_num_categories) via build_encoder_input_dim: r2c2→9, r2c3→14, r4c4→72,
# r5c5→186. A checkpoint can ONLY be loaded at the geometry it was trained at, and
# the filename does not encode it, so every stage passes these explicitly.
MC_MAX_REG=4; MC_MAX_CAT=4

# ── Derived checkpoint paths ───────────────────────────────────────────────────
# The directory tag must agree with the filename's _b/_e fields; both derive from
# MC_TRAIN_BATCH / MC_EPOCHS so they cannot drift. The filename does not encode
# r/c, which is why the tag carries it.
MC_TAG="r${MC_MAX_REG}c${MC_MAX_CAT}_b${MC_TRAIN_BATCH}_e${MC_EPOCHS}"
MC_CKPT_DIR="./outputs/checkpoints/model_class/cf/${MC_TAG}"
MC_CKPT="${MC_CKPT_DIR}/cogformer_model_class_fm_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${MC_DIM}_b${MC_TRAIN_BATCH}_e${MC_EPOCHS}_t${STEPS}.pt"

# LBA_FIXED_A is the value every LBA arm must use -- finetune, zero-shot and
# validation alike. It is in the checkpoint filename because fixed_A changes what
# the intrinsic `a` MEANS (threshold vs start-point range), so runs at different
# values are not comparable.
LBA_FIXED_A=0.5
# 128, matching the model-class and BayesFlow batch (decided 2026-09-28; was 256).
LBA_FT_BATCH=128
LBA_FT_CKPT="./outputs/checkpoints/model_class/cf/lba_finetune/cogformer_lba_finetune_s2_A${LBA_FIXED_A}_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${MC_DIM}_b${LBA_FT_BATCH}_e${LBA_FT_EPOCHS}_t${STEPS}.pt"
