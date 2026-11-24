#!/usr/bin/env bash
# shared_config.sh — hyperparameters and derived checkpoint paths for the finals
# pipeline (run_experiments.sh). Source this file; do not execute it directly.
# experiments/shared_config.py reads the same values for the Python assemblers.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STUDY_ROOT="${STUDY_ROOT:-${PROJECT_ROOT}/outputs}"
[[ "$STUDY_ROOT" == /* ]] || STUDY_ROOT="${PROJECT_ROOT}/${STUDY_ROOT}"
STUDY_ROOT="$(realpath -m "$STUDY_ROOT")"
export STUDY_ROOT
export RESULTS_ROOT="$STUDY_ROOT"

# ── Architecture ───────────────────────────────────────────────────────────────
# Encoder matched to the BayesFlow baselines' SetTransformer summary network (width 128,
# 4 heads of 32, 4 blocks; adopted 2026-10-03 from the arch A/B, logs/arch_ab.sh): it
# beat the 64/8/8 encoder on NRMSE in 237/244 parameter rows at equal budget. Head
# count changes no parameter shape, so a loader built with the wrong ENC_HEADS loads
# the checkpoint WITHOUT error -- every stage must pass these.
ENC_LAYERS=4
ENC_HEADS=4
ENC_EMBED=128
DEC_LAYERS=8
HEADS=8      # decoder heads (the _h field of the checkpoint filenames)
PROJ=256
SEEDS=32
MC_DIM=128   # seed_dim of the model-class checkpoint
# Flow-matching training-time power law, t = u ** (1 + alpha): BayesFlow's default 0.5,
# which every BF baseline trains with, so both gap arms optimise the same objective
# (adopted 2026-10-03; it was 0, uniform). Training only: no effect on loading/sampling.
FM_TIME_ALPHA=0.5
# Posterior sampler for every CogFormer scoring stage. The BF baselines sample with
# BayesFlow's default adaptive Tsit5 (5th order, min_steps=50, rtol 1e-4, atol 1e-6),
# i.e. >= 50 steps and >= 300 network evaluations. The closest fixed-step solver here
# is RK4 on the same 50-step floor (200 evaluations; was midpoint x 50 = 100, and
# euler in the ensemble and design-EIG stages). Adopted 2026-10-03.
FM_SOLVER=rk4
FM_SAMPLE_STEPS=50
# Posterior draws per sampling pass. At 1 the GPU is launch-bound on the 28-token bench5
# cases: RK4 x 50 took 109 s per 200x200 case at chunk 1 and 56 s at chunk 5 (midpoint
# at chunk 1, the old setting, 67 s). Large designs (~150+ tokens: mixed_predictor,
# ensemble, real-data designs) are compute-bound already at 1, so 5 is neutral there.
# Large designs would pay in memory for nothing (576 tokens: 11.9 GiB at 5 vs 2.7 at 1),
# so chunked_sample caps it by tokens per pass (experiments/utils.py
# MAX_SAMPLE_TOKENS_PER_PASS): 5 up to 150 tokens, 1 from ~375 (at B=200).
FM_SAMPLE_CHUNK=5

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

# Simulation backend for EVERY stage (exported as COGFORMER_SIM_BACKEND by
# run_experiments.sh). "gpu" runs all five families (DDM/RDM/CDM/LBA/aDDM) on Triton
# kernels, distributionally equivalent to numba (experiments/check_gpu_simulators.py),
# 95-420x one CPU core. Seeded outputs reproduce only on the backend that made them, so
# do not switch it mid-study; "numba" is the CPU reference implementation.
SIM_BACKEND=gpu

# ── Design generator (study identity) ──────────────────────────────────────────
# The default study can start from a source-only checkout.
STUDY_CONFIG="$STUDY_ROOT/study.env"
if [[ ! -f "$STUDY_CONFIG" && "$STUDY_ROOT" == "$PROJECT_ROOT/outputs" ]]; then
    STUDY_CONFIG="$PROJECT_ROOT/experiments/paper_study.env"
fi
if [[ -f "$STUDY_CONFIG" ]]; then source "$STUDY_CONFIG"; fi
DESIGN_GENERATOR="${DESIGN_GENERATOR:-mixture}"
GEN_SUFFIX=""; [[ "$DESIGN_GENERATOR" == mixture ]] || GEN_SUFFIX="_${DESIGN_GENERATOR}"

# Study-local structural prior and encoder mode. Defaults retain legacy studies.
INTERACTION_POLICY="${INTERACTION_POLICY:-all}"
INTERACTION_PROB="${INTERACTION_PROB:-}"
ENCODER_STRUCTURE_CONDITIONING="${ENCODER_STRUCTURE_CONDITIONING:-false}"
case "$INTERACTION_POLICY" in all|optional) ;; *) echo "invalid INTERACTION_POLICY" >&2; return 2 ;; esac
case "$ENCODER_STRUCTURE_CONDITIONING" in true|false) ;; *) echo "invalid ENCODER_STRUCTURE_CONDITIONING" >&2; return 2 ;; esac
STRUCTURE_SUFFIX=""
if [[ "$INTERACTION_POLICY" == optional ]]; then
    PROB_TAG=beta22
    if [[ -n "$INTERACTION_PROB" ]]; then
        PROB_TAG="$("${PY:-python}" -c 'import math,sys; p=float(sys.argv[1]); assert math.isfinite(p) and 0 <= p <= 1, "INTERACTION_PROB must be in [0,1]"; print(format(p, "g"))' "$INTERACTION_PROB")" || return 2
    fi
    STRUCTURE_SUFFIX="_intoptional_${PROB_TAG}"
elif [[ -n "$INTERACTION_PROB" ]]; then
    echo "INTERACTION_PROB requires INTERACTION_POLICY=optional" >&2; return 2
fi
[[ "$ENCODER_STRUCTURE_CONDITIONING" == false ]] || STRUCTURE_SUFFIX+="_encstruct"
# Explicit entity selection wins over the W&B login's default team.
export WANDB_ENTITY="${WANDB_ENTITY:-jerrymhuang-rensselaer-polytechnic-institute}"

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
# The encoder and alpha fields keep a checkpoint from an earlier architecture (whose
# .done would otherwise satisfy e1_train) from being picked up under the same name.
# _ptc: trained after ModelClass.collate kept per_trial_context (2026-10-05); every
# earlier checkpoint saw favored_idx = 0 for all RDM/LBA training data.
# GEN_SUFFIX is empty for the mixture generator, so the main study's tag is unchanged.
MC_TAG="r${MC_MAX_REG}c${MC_MAX_CAT}_enc${ENC_EMBED}h${ENC_HEADS}l${ENC_LAYERS}_a${FM_TIME_ALPHA}_b${MC_TRAIN_BATCH}_e${MC_EPOCHS}_ptc${GEN_SUFFIX}${STRUCTURE_SUFFIX}"
MC_CKPT_DIR="${RESULTS_ROOT}/checkpoints/model_class/cf/${MC_TAG}"
MC_CKPT="${MC_CKPT_DIR}/cogformer_model_class_fm_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${MC_DIM}_b${MC_TRAIN_BATCH}_e${MC_EPOCHS}_t${STEPS}${STRUCTURE_SUFFIX}.pt"

# LBA_FIXED_A is the value every LBA arm must use -- finetune, zero-shot and
# validation alike. It is in the checkpoint filename because fixed_A changes what
# the intrinsic `a` MEANS (threshold vs start-point range), so runs at different
# values are not comparable.
LBA_FIXED_A=0.5
# 128, matching the model-class and BayesFlow batch (decided 2026-09-28; was 256).
LBA_FT_BATCH=128
# Under MC_TAG: the finetune's filename does not encode its base checkpoint.
LBA_FT_CKPT="${RESULTS_ROOT}/checkpoints/model_class/cf/lba_finetune/${MC_TAG}/cogformer_lba_finetune_s2_A${LBA_FIXED_A}_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${MC_DIM}_b${LBA_FT_BATCH}_e${LBA_FT_EPOCHS}_t${STEPS}${STRUCTURE_SUFFIX}.pt"

# Track every CogFormer training (e1_train, e3_finetune) in wandb project
# "cogformer-model-class". The orchestrator refuses to start without credentials
# (`wandb login`, or WANDB_API_KEY); WANDB_MODE=offline logs locally for a later
# `wandb sync`. Override per launch with USE_WANDB=0.
TRACK_WANDB=1
