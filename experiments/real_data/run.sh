#!/usr/bin/env bash
# real_data/run.sh — Apply trained checkpoints to real data (DDM mf/mc, CDM mf/mc).
#
# Depends on: DDM_MF_CKPT, CDM_MF_CKPT, MC_CKPT (produced by model_family/run.sh
# and model_class/run.sh).

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../shared_config.sh"

echo "=== Real data: DDM model-family apply ==="
python -m experiments.real_data.ddm_mf_cf_apply \
    --checkpoint "$DDM_MF_CKPT" \
    --encoder_num_layers "$ENC_LAYERS" \
    --decoder_num_layers "$DEC_LAYERS" \
    --encoder_num_heads  "$HEADS" \
    --decoder_num_heads  "$HEADS" \
    --proj_dim  "$PROJ" \
    --num_seeds "$SEEDS" \
    --seed_dim  "$DIM"

echo "=== Real data: DDM model-class apply ==="
python -m experiments.real_data.ddm_mc_cf_apply \
    --checkpoint "$MC_CKPT" \
    --encoder_num_layers "$ENC_LAYERS" \
    --decoder_num_layers "$DEC_LAYERS" \
    --encoder_num_heads  "$HEADS" \
    --decoder_num_heads  "$HEADS" \
    --proj_dim      "$PROJ" \
    --num_seeds     "$SEEDS" \
    --seed_dim      "$MC_DIM" \
    --model_embed_dim 8

echo "=== Real data: CDM model-family apply ==="
python -m experiments.real_data.cdm_mf_cf_apply \
    --checkpoint "$CDM_MF_CKPT" \
    --encoder_num_layers "$ENC_LAYERS" \
    --decoder_num_layers "$DEC_LAYERS" \
    --encoder_num_heads  "$HEADS" \
    --decoder_num_heads  "$HEADS" \
    --proj_dim  "$PROJ" \
    --num_seeds "$SEEDS" \
    --seed_dim  "$DIM"

echo "=== Real data: CDM model-class apply ==="
python -m experiments.real_data.cdm_mc_cf_apply \
    --checkpoint "$MC_CKPT" \
    --encoder_num_layers "$ENC_LAYERS" \
    --decoder_num_layers "$DEC_LAYERS" \
    --encoder_num_heads  "$HEADS" \
    --decoder_num_heads  "$HEADS" \
    --proj_dim      "$PROJ" \
    --num_seeds     "$SEEDS" \
    --seed_dim      "$MC_DIM" \
    --model_embed_dim 8
