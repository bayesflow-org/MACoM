#!/usr/bin/env bash
# run_all.sh — sequential orchestration of the full CogFormer experiment suite.
#
# 9-step order (agreed 2026-05-02):
#   1. model_family_cf_train (DDM, CDM, RDM) → validate
#   2. model_class_cf_train → validate
#   3. model_class_lba_cf_validate
#   4. real_data apply (DDM mf/mc, CDM mf/mc)
#   5. generalization_cf_train (DDM × 4 holdouts) → validate
#   6. encoder_rep_cf_train → lca_bf_apply → ffi_bf_apply
#   7. Ablations (A–F run_ablations.sh, then G scaling_study)
#   8. model_family_bf_train (DDM, CDM, RDM) → validate
#   9. model_family_c2st
#
# Requirements:
#   - 1 GPU; all steps run sequentially
#   - Fill in the CKPT_* path variables before running Steps 2-9 that
#     depend on earlier checkpoints; names are constructed automatically
#     from the shared hyper variables below.
#
# Usage:
#   bash experiments/run_all.sh 2>&1 | tee outputs/logs/run_all.log

set -euo pipefail
mkdir -p outputs/logs

# ── Shared hyperparameters ─────────────────────────────────────────────────────
ENC_LAYERS=8
DEC_LAYERS=8
HEADS=8
PROJ=256
SEEDS=32
DIM=64        # seed_dim for model-family runs
MC_DIM=128    # seed_dim for model-class / encoder-rep runs
OBS=500
MIN_OBS=200
MAX_OBS=1000
BATCH=64
MC_BATCH=64
VAL_BATCH=500
EPOCHS=5000
STEPS=100

MF_BF_EPOCHS=1000   # BayesFlow training epochs

# ── Derived checkpoint filenames (auto-constructed) ────────────────────────────

# model_family_cf_train checkpoints
ddm_mf_ckpt_name="cogformer_mixed_attn_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${DIM}_o${OBS}_b${BATCH}_e${EPOCHS}_t${STEPS}.pt"
cdm_mf_ckpt_name="cogformer_cdm_mixed_attn_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${DIM}_o${OBS}_b${BATCH}_e${EPOCHS}_t${STEPS}.pt"
rdm_mf_ckpt_name="cogformer_rdm_mixed_attn_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${DIM}_o${OBS}_b${BATCH}_e${EPOCHS}_t${STEPS}.pt"
DDM_MF_CKPT="./outputs/checkpoints/model_family/cf/ddm/${ddm_mf_ckpt_name}"
CDM_MF_CKPT="./outputs/checkpoints/model_family/cf/cdm/${cdm_mf_ckpt_name}"
RDM_MF_CKPT="./outputs/checkpoints/model_family/cf/rdm/${rdm_mf_ckpt_name}"

# model_class_cf_train checkpoint
mc_ckpt_name="cogformer_model_class_fm_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${MC_DIM}_b${MC_BATCH}_e${EPOCHS}_t${STEPS}.pt"
MC_CKPT="./outputs/checkpoints/model_class/cf/${mc_ckpt_name}"

# encoder_rep_cf_train checkpoint
enc_rep_ckpt_name="cogformer_encoder_rep_fm_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${MC_DIM}_b${BATCH}_e${EPOCHS}_t${STEPS}.pt"
ENC_REP_CKPT="./outputs/checkpoints/encoder_rep/cf/${enc_rep_ckpt_name}"

# ── Helper functions ───────────────────────────────────────────────────────────

run_mf_train() {
    local model="$1"
    python -m experiments.model_family.model_family_cf_train \
        --model "$model" \
        --encoder_num_layers "$ENC_LAYERS" \
        --decoder_num_layers "$DEC_LAYERS" \
        --encoder_num_heads "$HEADS" \
        --decoder_num_heads "$HEADS" \
        --proj_dim "$PROJ" \
        --num_seeds "$SEEDS" \
        --seed_dim "$DIM" \
        --num_obs "$OBS" \
        --min_num_obs "$MIN_OBS" \
        --max_num_obs "$MAX_OBS" \
        --train_batch_size "$BATCH" \
        --val_batch_size "$VAL_BATCH" \
        --epochs "$EPOCHS" \
        --steps_per_epoch "$STEPS" \
        --lr 1e-4 \
        --dropout 0.05 \
        --layer_dropout 0.05 \
        --fm_sample_steps 50 \
        --fm_num_samples 200 \
        --fm_solver midpoint
}

run_mf_validate() {
    local model="$1"
    local ckpt="$2"
    python -m experiments.model_family.model_family_cf_validate \
        --model "$model" \
        --checkpoint "$ckpt" \
        --encoder_num_layers "$ENC_LAYERS" \
        --decoder_num_layers "$DEC_LAYERS" \
        --encoder_num_heads "$HEADS" \
        --decoder_num_heads "$HEADS" \
        --proj_dim "$PROJ" \
        --num_seeds "$SEEDS" \
        --seed_dim "$DIM" \
        --batch_size 200 \
        --num_obs "$OBS"
}

run_mc_train() {
    python -m experiments.model_class.model_class_cf_train \
        --encoder_num_layers "$ENC_LAYERS" \
        --decoder_num_layers "$DEC_LAYERS" \
        --encoder_num_heads "$HEADS" \
        --decoder_num_heads "$HEADS" \
        --proj_dim "$PROJ" \
        --num_seeds "$SEEDS" \
        --seed_dim "$MC_DIM" \
        --model_embed_dim 8 \
        --min_num_obs "$MIN_OBS" \
        --max_num_obs "$MAX_OBS" \
        --train_batch_size "$MC_BATCH" \
        --val_batch_size "$VAL_BATCH" \
        --epochs "$EPOCHS" \
        --steps_per_epoch "$STEPS" \
        --lr 1e-4 \
        --dropout 0.05 \
        --layer_dropout 0.05 \
        --fm_sample_steps 50 \
        --fm_num_samples 100
}

run_gen_train() {
    local model="$1"
    local holdout="$2"
    python -m experiments.generalization.generalization_cf_train \
        --model "$model" \
        --holdout_type "$holdout" \
        --encoder_num_layers "$ENC_LAYERS" \
        --decoder_num_layers "$DEC_LAYERS" \
        --encoder_num_heads "$HEADS" \
        --decoder_num_heads "$HEADS" \
        --proj_dim "$PROJ" \
        --num_seeds "$SEEDS" \
        --seed_dim "$DIM" \
        --num_obs "$OBS" \
        --min_num_obs "$MIN_OBS" \
        --max_num_obs "$MAX_OBS" \
        --train_batch_size "$BATCH" \
        --val_batch_size "$VAL_BATCH" \
        --epochs "$EPOCHS" \
        --steps_per_epoch "$STEPS" \
        --val_epoch_interval 1000 \
        --lr 1e-4 \
        --dropout 0.05 \
        --layer_dropout 0.05 \
        --fm_sample_steps 50 \
        --fm_num_samples 200 \
        --fm_solver midpoint
}

# ── Step 1: ModelFamily CF train + validate ────────────────────────────────────
echo "=== Step 1: ModelFamily CF (DDM, CDM, RDM) ==="

for model in ddm cdm rdm; do
    echo "  Training $model ..."
    run_mf_train "$model"
done

echo "  Validating DDM ..."
run_mf_validate ddm "$DDM_MF_CKPT"
echo "  Validating CDM ..."
run_mf_validate cdm "$CDM_MF_CKPT"
echo "  Validating RDM ..."
run_mf_validate rdm "$RDM_MF_CKPT"

# ── Step 2: ModelClass CF train + validate ─────────────────────────────────────
echo "=== Step 2: ModelClass CF ==="
run_mc_train

python -m experiments.model_class.model_class_cf_validate \
    --checkpoint "$MC_CKPT" \
    --encoder_num_layers "$ENC_LAYERS" \
    --decoder_num_layers "$DEC_LAYERS" \
    --encoder_num_heads "$HEADS" \
    --decoder_num_heads "$HEADS" \
    --proj_dim "$PROJ" \
    --num_seeds "$SEEDS" \
    --seed_dim "$MC_DIM" \
    --model_embed_dim 8 \
    --batch_size 200 \
    --num_obs "$OBS"

# ── Step 3: ModelClass LBA validate ───────────────────────────────────────────
echo "=== Step 3: ModelClass LBA validate ==="
python -m experiments.generalization.model_class_lba_cf_validate \
    --checkpoint "$MC_CKPT" \
    --encoder_num_layers "$ENC_LAYERS" \
    --decoder_num_layers "$DEC_LAYERS" \
    --encoder_num_heads "$HEADS" \
    --decoder_num_heads "$HEADS" \
    --proj_dim "$PROJ" \
    --num_seeds "$SEEDS" \
    --seed_dim "$MC_DIM" \
    --model_embed_dim 8 \
    --batch_size 200 \
    --num_obs "$OBS"

# ── Step 4: Real data apply (DDM mf/mc, CDM mf/mc) ───────────────────────────
echo "=== Step 4: Real data apply ==="
python -m experiments.real_data.ddm_mf_cf_apply \
    --checkpoint_path "$DDM_MF_CKPT" \
    --encoder_num_layers "$ENC_LAYERS" \
    --decoder_num_layers "$DEC_LAYERS" \
    --encoder_num_heads "$HEADS" \
    --decoder_num_heads "$HEADS" \
    --proj_dim "$PROJ" \
    --num_seeds "$SEEDS" \
    --seed_dim "$DIM"

python -m experiments.real_data.ddm_mc_cf_apply \
    --checkpoint_path "$MC_CKPT" \
    --encoder_num_layers "$ENC_LAYERS" \
    --decoder_num_layers "$DEC_LAYERS" \
    --encoder_num_heads "$HEADS" \
    --decoder_num_heads "$HEADS" \
    --proj_dim "$PROJ" \
    --num_seeds "$SEEDS" \
    --seed_dim "$MC_DIM" \
    --model_embed_dim 8

python -m experiments.real_data.cdm_mf_cf_apply \
    --checkpoint_path "$CDM_MF_CKPT" \
    --encoder_num_layers "$ENC_LAYERS" \
    --decoder_num_layers "$DEC_LAYERS" \
    --encoder_num_heads "$HEADS" \
    --decoder_num_heads "$HEADS" \
    --proj_dim "$PROJ" \
    --num_seeds "$SEEDS" \
    --seed_dim "$DIM"

python -m experiments.real_data.cdm_mc_cf_apply \
    --checkpoint_path "$MC_CKPT" \
    --encoder_num_layers "$ENC_LAYERS" \
    --decoder_num_layers "$DEC_LAYERS" \
    --encoder_num_heads "$HEADS" \
    --decoder_num_heads "$HEADS" \
    --proj_dim "$PROJ" \
    --num_seeds "$SEEDS" \
    --seed_dim "$MC_DIM" \
    --model_embed_dim 8

# ── Step 5: Generalization CF train + validate ─────────────────────────────────
echo "=== Step 5: Generalization CF (DDM × 4 holdout types) ==="
for holdout in random variability interaction sparsity; do
    echo "  gen ddm/$holdout ..."
    run_gen_train "ddm" "$holdout"
done

# Validation: construct gen checkpoint paths and validate each
echo "  Validating generalization ..."
prefix="cogformer_mixed_attn"
for holdout in random variability interaction sparsity; do
    gen_ckpt="./outputs/checkpoints/generalization/cf/ddm/${prefix}_gen_${holdout}_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${DIM}_o${OBS}_b${BATCH}_e${EPOCHS}_t${STEPS}.pt"
    python -m experiments.generalization.generalization_cf_validate \
        --model ddm \
        --holdout_type "$holdout" \
        --checkpoint "$gen_ckpt" \
        --encoder_num_layers "$ENC_LAYERS" \
        --decoder_num_layers "$DEC_LAYERS" \
        --encoder_num_heads "$HEADS" \
        --decoder_num_heads "$HEADS" \
        --proj_dim "$PROJ" \
        --num_seeds "$SEEDS" \
        --seed_dim "$DIM" \
        --batch_size 200 \
        --num_obs "$OBS"
done

# ── Step 6: Encoder rep CF train → LCA/FFI BF apply ───────────────────────────
echo "=== Step 6: Encoder Rep CF + LCA/FFI BF ==="

python -m experiments.encoder_rep.encoder_rep_cf_train \
    --encoder_num_layers "$ENC_LAYERS" \
    --decoder_num_layers "$DEC_LAYERS" \
    --encoder_num_heads "$HEADS" \
    --decoder_num_heads "$HEADS" \
    --proj_dim "$PROJ" \
    --num_seeds "$SEEDS" \
    --seed_dim "$MC_DIM" \
    --model_embed_dim 8 \
    --min_num_obs "$MIN_OBS" \
    --max_num_obs "$MAX_OBS" \
    --train_batch_size "$BATCH" \
    --val_batch_size "$VAL_BATCH" \
    --epochs "$EPOCHS" \
    --steps_per_epoch "$STEPS" \
    --lr 1e-4 \
    --dropout 0.05 \
    --layer_dropout 0.05 \
    --fm_sample_steps 50 \
    --fm_num_samples 100

for case in intercept_only intercept_v; do
    echo "  LCA BF $case train ..."
    python -m experiments.encoder_rep.lca_bf_apply \
        --mode train \
        --case "$case" \
        --cogformer_checkpoint "$ENC_REP_CKPT" \
        --encoder_num_layers "$ENC_LAYERS" \
        --decoder_num_layers "$DEC_LAYERS" \
        --encoder_num_heads "$HEADS" \
        --decoder_num_heads "$HEADS" \
        --num_seeds "$SEEDS" \
        --seed_dim "$MC_DIM" \
        --proj_dim "$PROJ" \
        --model_embed_dim 8 \
        --epochs 500 \
        --steps_per_epoch 100

    echo "  LCA BF $case validate ..."
    python -m experiments.encoder_rep.lca_bf_apply \
        --mode validate \
        --case "$case" \
        --cogformer_checkpoint "$ENC_REP_CKPT" \
        --encoder_num_layers "$ENC_LAYERS" \
        --decoder_num_layers "$DEC_LAYERS" \
        --encoder_num_heads "$HEADS" \
        --decoder_num_heads "$HEADS" \
        --num_seeds "$SEEDS" \
        --seed_dim "$MC_DIM" \
        --proj_dim "$PROJ" \
        --model_embed_dim 8

    echo "  FFI BF $case train ..."
    python -m experiments.encoder_rep.ffi_bf_apply \
        --mode train \
        --case "$case" \
        --cogformer_checkpoint "$ENC_REP_CKPT" \
        --encoder_num_layers "$ENC_LAYERS" \
        --decoder_num_layers "$DEC_LAYERS" \
        --encoder_num_heads "$HEADS" \
        --decoder_num_heads "$HEADS" \
        --num_seeds "$SEEDS" \
        --seed_dim "$MC_DIM" \
        --proj_dim "$PROJ" \
        --model_embed_dim 8 \
        --epochs 500 \
        --steps_per_epoch 100

    echo "  FFI BF $case validate ..."
    python -m experiments.encoder_rep.ffi_bf_apply \
        --mode validate \
        --case "$case" \
        --cogformer_checkpoint "$ENC_REP_CKPT" \
        --encoder_num_layers "$ENC_LAYERS" \
        --decoder_num_layers "$DEC_LAYERS" \
        --encoder_num_heads "$HEADS" \
        --decoder_num_heads "$HEADS" \
        --num_seeds "$SEEDS" \
        --seed_dim "$MC_DIM" \
        --proj_dim "$PROJ" \
        --model_embed_dim 8
done

# ── Step 7: Ablations (A–F), then scaling study (G) ───────────────────────────
echo "=== Step 7: Ablations ==="
bash experiments/ablations/run_ablations.sh

# G: scaling study — run after ablation training completes.
# Replace <CKPT> with the actual DDM model-family checkpoint.
# echo "  G: Dimensionality scaling study ..."
# python -m experiments.ablations.scaling_study \
#     --model ddm \
#     --checkpoint_path "$DDM_MF_CKPT" \
#     --output_dir ./outputs/evaluations/ablations/scaling_study/ddm/ \
#     --regressor_counts 0 1 2 3 4 5 \
#     --n_eval_batches 10 \
#     --eval_batch_size 200 \
#     --num_obs "$OBS" \
#     --fm_sample_steps 50 \
#     --fm_num_samples 200 \
#     --fm_solver rk4

# ── Step 8: ModelFamily BF train + validate ────────────────────────────────────
echo "=== Step 8: ModelFamily BF (DDM, CDM, RDM) ==="
BF_CASES=(intercept_only fixed regressed)
for model in ddm cdm rdm; do
    for case in "${BF_CASES[@]}"; do
        echo "  BF train $model/$case ..."
        python -m experiments.model_family.model_family_bf_train \
            --model "$model" \
            --case "$case" \
            --epochs "$MF_BF_EPOCHS" \
            --steps_per_epoch 100 \
            --batch_size 64
    done
done

echo "  BF validate ..."
for model in ddm cdm rdm; do
    python -m experiments.model_family.model_family_bf_validate \
        --model "$model"
done

# ── Step 9: C2ST ───────────────────────────────────────────────────────────────
echo "=== Step 9: C2ST ==="
for model in ddm cdm rdm; do
    python -m experiments.model_family.model_family_c2st \
        --model "$model"
done

echo "=== All experiments complete. ==="
python -m experiments.timing_report
