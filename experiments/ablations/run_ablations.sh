#!/usr/bin/env bash
# run_ablations.sh — launches all CogFormer ablation experiments.
#
# Each section corresponds to one ablation study. Sections can be run independently
# by commenting out the others. All model-family ablations use DDM as the reference
# model; extend to CDM/RDM by duplicating runs with --model cdm / --model rdm.
#
# Usage:
#   bash experiments/ablations/run_ablations.sh
#   python -m experiments.ablations.run_ablations  # not applicable; run as bash script

set -euo pipefail

MF=experiments.ablations.ablation_mf_train
MC=experiments.ablations.ablation_mc_train

# If DRY_EPOCHS is set in the environment (e.g. exported from dry_run.sh),
# forward it to every training invocation as --dry_run_epochs.
DRY_OPTS=()
if [[ -n "${DRY_EPOCHS:-}" ]]; then
    DRY_OPTS=(--dry_run_epochs "$DRY_EPOCHS")
fi

# -----------------------------------------------------------------------
# Shared baseline hyperparameters (model-family ablations)
# -----------------------------------------------------------------------
run_mf() {
    python -m "$MF" \
        --model ddm \
        --epochs 5000 \
        --steps_per_epoch 100 \
        --train_batch_size 64 \
        --val_batch_size 500 \
        --num_obs 500 \
        --min_num_obs 200 \
        --max_num_obs 1000 \
        --encoder_num_layers 8 \
        --decoder_num_layers 8 \
        --encoder_num_heads 8 \
        --decoder_num_heads 8 \
        --proj_dim 256 \
        --num_seeds 32 \
        --seed_dim 64 \
        --time_embedding_dim 32 \
        --pos_embedding_dim 32 \
        --dropout 0.05 \
        --layer_dropout 0.05 \
        --lr 1e-4 \
        --fm_sample_steps 50 \
        --fm_num_samples 200 \
        --fm_solver midpoint \
        "${DRY_OPTS[@]}" \
        "$@"
}

# Shared baseline hyperparameters (model-class ablation)
run_mc() {
    python -m "$MC" \
        "${DRY_OPTS[@]}" \
        --epochs 5000 \
        --steps_per_epoch 100 \
        --train_batch_size 64 \
        --val_batch_size 500 \
        --min_num_obs 200 \
        --max_num_obs 1000 \
        --val_num_obs 500 \
        --encoder_num_layers 8 \
        --decoder_num_layers 8 \
        --encoder_num_heads 8 \
        --decoder_num_heads 8 \
        --proj_dim 256 \
        --num_seeds 32 \
        --seed_dim 128 \
        --model_embed_dim 8 \
        --time_embedding_dim 32 \
        --pos_embedding_dim 32 \
        --dropout 0.05 \
        --layer_dropout 0.05 \
        --lr 1e-4 \
        --fm_sample_steps 50 \
        --fm_num_samples 100 \
        "$@"
}

# -----------------------------------------------------------------------
# A. PMA Bottleneck: num_seeds ∈ {1, 4, 8, 16, 32, 64}
#    Ablates the information capacity of the encoder's pooling bottleneck.
#    Baseline: num_seeds=32.
# -----------------------------------------------------------------------
echo "=== A. PMA Seeds ==="
for seeds in 1 4 8 16 32 64; do
    echo "  num_seeds=$seeds"
    run_mf \
        --num_seeds "$seeds" \
        --checkpoint_dir "./outputs/checkpoints/ablations/pma_seeds/ddm/" \
        --checkpoint_prefix "abl_pma" \
        --fig_subdir "ablations/pma_seeds/ddm"
done

# -----------------------------------------------------------------------
# B. Decoder Attention Pattern
#    Ablates the layer interleaving strategy in the decoder.
#    Conditions:
#      B1 mixed_attention, mab_first=True  (baseline)
#      B2 mixed_attention, mab_first=False (SAB before MAB)
#      B3 self_attention only              (no encoder cross-attention)
#      B4 cross_attention only             (no intra-parameter self-attention)
# -----------------------------------------------------------------------
echo "=== B. Decoder Attention Pattern ==="

echo "  B1: mixed, MAB-first (baseline)"
run_mf \
    --decoder_layer_design mixed_attention --mab_first \
    --checkpoint_dir "./outputs/checkpoints/ablations/attn_pattern/ddm/" \
    --checkpoint_prefix "abl_attn" \
    --fig_subdir "ablations/attn_pattern/ddm"

echo "  B2: mixed, SAB-first"
run_mf \
    --decoder_layer_design mixed_attention --no-mab_first \
    --checkpoint_dir "./outputs/checkpoints/ablations/attn_pattern/ddm/" \
    --checkpoint_prefix "abl_attn" \
    --fig_subdir "ablations/attn_pattern/ddm"

echo "  B3: self-attention only"
run_mf \
    --decoder_layer_design self_attention \
    --checkpoint_dir "./outputs/checkpoints/ablations/attn_pattern/ddm/" \
    --checkpoint_prefix "abl_attn" \
    --fig_subdir "ablations/attn_pattern/ddm"

echo "  B4: cross-attention only"
run_mf \
    --decoder_layer_design cross_attention \
    --checkpoint_dir "./outputs/checkpoints/ablations/attn_pattern/ddm/" \
    --checkpoint_prefix "abl_attn" \
    --fig_subdir "ablations/attn_pattern/ddm"

# -----------------------------------------------------------------------
# C. Time Conditioning: Output-only FiLM vs. Block-level FiLM
#    Ablates whether time modulation is applied only at the output head
#    (current design) or after every attention block as well.
#    C1 output-only FiLM  (baseline)
#    C2 block-level FiLM  (FiLM at every layer + output)
# -----------------------------------------------------------------------
echo "=== C. Block-level FiLM ==="

echo "  C1: output-only FiLM (baseline)"
run_mf \
    --checkpoint_dir "./outputs/checkpoints/ablations/block_film/ddm/" \
    --checkpoint_prefix "abl_film" \
    --fig_subdir "ablations/block_film/ddm"

echo "  C2: block-level FiLM"
run_mf \
    --block_film \
    --checkpoint_dir "./outputs/checkpoints/ablations/block_film/ddm/" \
    --checkpoint_prefix "abl_film" \
    --fig_subdir "ablations/block_film/ddm"

# -----------------------------------------------------------------------
# D. Fourier Frequency Trainability
#    Ablates whether the time embedding's Fourier frequencies are learned
#    or fixed at initialisation.
#    D1 trainable frequencies  (baseline)
#    D2 fixed frequencies
# -----------------------------------------------------------------------
echo "=== D. Fourier Frequency Trainability ==="

echo "  D1: trainable frequencies (baseline)"
run_mf \
    --time_freq_trainable \
    --checkpoint_dir "./outputs/checkpoints/ablations/freq_trainable/ddm/" \
    --checkpoint_prefix "abl_freq" \
    --fig_subdir "ablations/freq_trainable/ddm"

echo "  D2: fixed frequencies"
run_mf \
    --no-time_freq_trainable \
    --checkpoint_dir "./outputs/checkpoints/ablations/freq_trainable/ddm/" \
    --checkpoint_prefix "abl_freq" \
    --fig_subdir "ablations/freq_trainable/ddm"

# -----------------------------------------------------------------------
# E. Encoder / Decoder Depth Split (total layers fixed at 16)
#    Ablates how capacity is distributed between encoder and decoder.
#    All conditions share the same total layer count as the baseline (8+8).
#    E1  2 enc + 14 dec  (minimal encoder)
#    E2  4 enc + 12 dec
#    E3  8 enc +  8 dec  (baseline, symmetric)
#    E4 12 enc +  4 dec
#    E5 14 enc +  2 dec  (minimal decoder)
# -----------------------------------------------------------------------
echo "=== E. Encoder / Decoder Depth Split (total = 16) ==="

declare -a ENC_LAYERS=(2  4  8  12 14)
declare -a DEC_LAYERS=(14 12  8   4  2)

for i in "${!ENC_LAYERS[@]}"; do
    enc="${ENC_LAYERS[$i]}"
    dec="${DEC_LAYERS[$i]}"
    echo "  E$((i+1)): enc=$enc, dec=$dec"
    run_mf \
        --encoder_num_layers "$enc" \
        --decoder_num_layers "$dec" \
        --checkpoint_dir "./outputs/checkpoints/ablations/depth_split/ddm/" \
        --checkpoint_prefix "abl_depth" \
        --fig_subdir "ablations/depth_split/ddm"
done

# -----------------------------------------------------------------------
# F. Model Embedding (ModelClass: DDM + CDM + RDM)
#    Ablates whether the network receives an explicit model-identity
#    embedding when trained on multiple SSMs simultaneously.
#    F1 with model embedding    (baseline, model_embed_dim=8)
#    F2 without model embedding (num_models=None)
# -----------------------------------------------------------------------
echo "=== F. Model Embedding (ModelClass) ==="

echo "  F1: with model embedding (baseline)"
run_mc \
    --checkpoint_dir "./outputs/checkpoints/ablations/model_embedding/" \
    --checkpoint_prefix "abl_mc"

echo "  F2: without model embedding"
run_mc \
    --no_model_embedding \
    --checkpoint_dir "./outputs/checkpoints/ablations/model_embedding/" \
    --checkpoint_prefix "abl_mc"

# -----------------------------------------------------------------------
# G. Dimensionality Scaling Study (2-iii)
#    Evaluates a trained checkpoint across designs with r = 0..5 regressors.
#    Measures whether recovery quality degrades as the parameter matrix grows.
#    This is a post-hoc evaluation — run AFTER training completes.
#    Replace <CKPT> with the actual checkpoint filename.
# -----------------------------------------------------------------------
# echo "=== G. Dimensionality Scaling Study ==="
# CKPT=./outputs/checkpoints/model_family/cf/ddm/<CKPT>.pt
# python -m experiments.ablations.scaling_study \
#     --model ddm \
#     --checkpoint_path "$CKPT" \
#     --output_dir ./outputs/scaling_study/ddm/ \
#     --regressor_counts 0 1 2 3 4 5 \
#     --n_eval_batches 10 \
#     --eval_batch_size 200 \
#     --num_obs 500 \
#     --fm_sample_steps 50 \
#     --fm_num_samples 200 \
#     --fm_solver rk4

echo "=== All ablation runs complete. ==="
