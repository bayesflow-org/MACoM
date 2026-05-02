#!/usr/bin/env bash
# dry_run.sh — end-to-end smoke test for the full CogFormer experiment suite.
#
# Runs every step in the agreed 9-step order, including checkpoint saving,
# validation, and downstream apply/analysis scripts. Training scripts use
# DRY_EPOCHS epochs so the total wall time approaches 24 hours; all other
# steps (validate, apply, c2st) run exactly as in run_all.sh.
#
# Steps that need an upstream checkpoint simply use the one saved during
# this dry run — the checkpoint filename is computed from --epochs (the full
# budget), not --dry_run_epochs, so the paths match automatically.
#
# There are ~53 training script invocations total. A good starting value for
# DRY_EPOCHS can be estimated as:
#   DRY_EPOCHS ≈ (target_hours × 3600) / (53 × steps_per_epoch × secs_per_step)
# where secs_per_step is measured from a short pilot (e.g. DRY_EPOCHS=2).
#
# Usage:
#   bash experiments/dry_run.sh [DRY_EPOCHS]
#
#   bash experiments/dry_run.sh 10   # explicit epoch count
#   bash experiments/dry_run.sh      # uses default below
#   bash experiments/dry_run.sh 2>&1 | tee outputs/logs/dry_run.log

set -euo pipefail
mkdir -p outputs/logs outputs/timing

# First positional argument overrides; default is 10.
# Tune this so the total wall time is ~24 hours.
export DRY_EPOCHS="${1:-10}"

# ── Shared hyperparameters (must match run_all.sh) ─────────────────────────────
ENC_LAYERS=8
DEC_LAYERS=8
HEADS=8
PROJ=256
SEEDS=32
DIM=64
MC_DIM=128
OBS=500
MIN_OBS=200
MAX_OBS=1000
BATCH=64
MC_BATCH=64
VAL_BATCH=500
EPOCHS=5000
STEPS=100

MF_BF_EPOCHS=1000
LCA_FFI_EPOCHS=500

# ── Derived checkpoint paths (identical to run_all.sh) ────────────────────────
ddm_mf_ckpt_name="cogformer_mixed_attn_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${DIM}_o${OBS}_b${BATCH}_e${EPOCHS}_t${STEPS}.pt"
cdm_mf_ckpt_name="cogformer_cdm_mixed_attn_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${DIM}_o${OBS}_b${BATCH}_e${EPOCHS}_t${STEPS}.pt"
rdm_mf_ckpt_name="cogformer_rdm_mixed_attn_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${DIM}_o${OBS}_b${BATCH}_e${EPOCHS}_t${STEPS}.pt"
DDM_MF_CKPT="./outputs/checkpoints/model_family/cf/ddm/${ddm_mf_ckpt_name}"
CDM_MF_CKPT="./outputs/checkpoints/model_family/cf/cdm/${cdm_mf_ckpt_name}"
RDM_MF_CKPT="./outputs/checkpoints/model_family/cf/rdm/${rdm_mf_ckpt_name}"

mc_ckpt_name="cogformer_model_class_fm_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${MC_DIM}_b${MC_BATCH}_e${EPOCHS}_t${STEPS}.pt"
MC_CKPT="./outputs/checkpoints/model_class/cf/${mc_ckpt_name}"

enc_rep_ckpt_name="cogformer_encoder_rep_fm_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${MC_DIM}_b${BATCH}_e${EPOCHS}_t${STEPS}.pt"
ENC_REP_CKPT="./outputs/checkpoints/encoder_rep/cf/${enc_rep_ckpt_name}"

echo "=== CogFormer end-to-end dry run: ${DRY_EPOCHS} epochs per training script (~53 invocations) ==="
echo "    Full-run budget: ${EPOCHS} epochs.  Extrapolation factor: $(echo "scale=1; $EPOCHS / $DRY_EPOCHS" | bc)×"
echo ""

# ── Step 1: ModelFamily CF train + validate ────────────────────────────────────
echo "=== Step 1: ModelFamily CF ==="

for model in ddm cdm rdm; do
    echo "  Training $model (dry) ..."
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
        --fm_solver midpoint \
        --dry_run_epochs "$DRY_EPOCHS"
done

echo "  Validating DDM ..."
python -m experiments.model_family.model_family_cf_validate \
    --model ddm \
    --checkpoint "$DDM_MF_CKPT" \
    --encoder_num_layers "$ENC_LAYERS" \
    --decoder_num_layers "$DEC_LAYERS" \
    --encoder_num_heads "$HEADS" \
    --decoder_num_heads "$HEADS" \
    --proj_dim "$PROJ" \
    --num_seeds "$SEEDS" \
    --seed_dim "$DIM" \
    --batch_size 200 \
    --num_obs "$OBS"

echo "  Validating CDM ..."
python -m experiments.model_family.model_family_cf_validate \
    --model cdm \
    --checkpoint "$CDM_MF_CKPT" \
    --encoder_num_layers "$ENC_LAYERS" \
    --decoder_num_layers "$DEC_LAYERS" \
    --encoder_num_heads "$HEADS" \
    --decoder_num_heads "$HEADS" \
    --proj_dim "$PROJ" \
    --num_seeds "$SEEDS" \
    --seed_dim "$DIM" \
    --batch_size 200 \
    --num_obs "$OBS"

echo "  Validating RDM ..."
python -m experiments.model_family.model_family_cf_validate \
    --model rdm \
    --checkpoint "$RDM_MF_CKPT" \
    --encoder_num_layers "$ENC_LAYERS" \
    --decoder_num_layers "$DEC_LAYERS" \
    --encoder_num_heads "$HEADS" \
    --decoder_num_heads "$HEADS" \
    --proj_dim "$PROJ" \
    --num_seeds "$SEEDS" \
    --seed_dim "$DIM" \
    --batch_size 200 \
    --num_obs "$OBS"

# ── Step 2: ModelClass CF train + validate ─────────────────────────────────────
echo "=== Step 2: ModelClass CF ==="

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
    --fm_num_samples 100 \
    --dry_run_epochs "$DRY_EPOCHS"

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

# ── Step 4: Real data apply ────────────────────────────────────────────────────
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
echo "=== Step 5: Generalization CF ==="

for model in ddm cdm rdm; do
    for holdout in random variability interaction sparsity; do
        echo "  gen $model/$holdout (dry) ..."
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
            --fm_solver midpoint \
            --dry_run_epochs "$DRY_EPOCHS"
    done
done

echo "  Validating generalization ..."
for model in ddm cdm rdm; do
    case "$model" in
        ddm) prefix="cogformer_mixed_attn" ;;
        cdm) prefix="cogformer_cdm_mixed_attn" ;;
        rdm) prefix="cogformer_rdm_mixed_attn" ;;
    esac
    for holdout in random variability interaction sparsity; do
        gen_ckpt="./outputs/checkpoints/generalization/cf/${model}/${prefix}_gen_${holdout}_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${DIM}_o${OBS}_b${BATCH}_e${EPOCHS}_t${STEPS}.pt"
        python -m experiments.generalization.generalization_cf_validate \
            --model "$model" \
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
done

# ── Step 6: Encoder Rep CF → LCA/FFI BF ───────────────────────────────────────
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
    --fm_num_samples 100 \
    --dry_run_epochs "$DRY_EPOCHS"

for case in intercept_only intercept_v; do
    echo "  LCA BF $case (train dry) ..."
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
        --epochs "$LCA_FFI_EPOCHS" \
        --steps_per_epoch 100 \
        --dry_run_epochs "$DRY_EPOCHS"

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

    echo "  FFI BF $case (train dry) ..."
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
        --epochs "$LCA_FFI_EPOCHS" \
        --steps_per_epoch 100 \
        --dry_run_epochs "$DRY_EPOCHS"

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

# ── Step 7: Ablations (A–F) + scaling study (G) ───────────────────────────────
echo "=== Step 7: Ablations ==="
# DRY_EPOCHS is already exported; run_ablations.sh picks it up automatically.
bash experiments/ablations/run_ablations.sh

echo "  G: Dimensionality scaling study ..."
python -m experiments.ablations.scaling_study \
    --model ddm \
    --checkpoint_path "$DDM_MF_CKPT" \
    --output_dir ./outputs/scaling_study/ddm/ \
    --regressor_counts 0 1 2 3 4 5 \
    --n_eval_batches 10 \
    --eval_batch_size 200 \
    --num_obs "$OBS" \
    --fm_sample_steps 50 \
    --fm_num_samples 200 \
    --fm_solver midpoint

# ── Step 8: ModelFamily BF train + validate ────────────────────────────────────
echo "=== Step 8: ModelFamily BF ==="
BF_CASES=(intercept_only fixed regressed)
for model in ddm cdm rdm; do
    for case in "${BF_CASES[@]}"; do
        echo "  BF train $model/$case (dry) ..."
        python -m experiments.model_family.model_family_bf_train \
            --model "$model" \
            --case "$case" \
            --epochs "$MF_BF_EPOCHS" \
            --steps_per_epoch 100 \
            --batch_size 64 \
            --dry_run_epochs "$DRY_EPOCHS"
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

# ── Timing report ──────────────────────────────────────────────────────────────
echo ""
echo "=== Timing report ==="
python -m experiments.timing_report

echo ""
echo "=== Dry run complete. All steps passed end-to-end. ==="
