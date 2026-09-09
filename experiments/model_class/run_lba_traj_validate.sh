#!/usr/bin/env bash
# Validate every LBA-finetune trajectory checkpoint on IDENTICAL datasets.
#
# --seed is the point. model_class_lba_finetune_validate calls seed_case(seed, family,
# scenario), which derives each case's RNG from the case's IDENTITY rather than its
# position, so all five checkpoints see the same simulated datasets. Without it, the
# epoch-to-epoch differences in the trajectory would be confounded with simulation
# noise. Seeding also forces serial simulation (num_workers=1), which is slower but is
# what makes the runs comparable.
#
# GEOMETRY: run_lba.sh omits --max_num_regressors/--max_num_categories and the script
# defaults to r5c5, which cannot load the r4c4 checkpoints. Passed explicitly here.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source experiments/shared_config.sh

SEED="${SEED:-0}"
# CHUNK does NOT buy speed here -- measured. chunk=1 and chunk=8 both sustain ~16
# posterior samples/min, i.e. the GPU is compute-saturated and chunking only moves the
# same work between kernel launches; chunk=25 simply OOMs (tries to allocate 27.3 GiB).
# So chunk stays small: identical throughput at ~5 GB instead of ~25 GB, leaving
# headroom for the heavier scenarios later in the sweep.
CHUNK="${CHUNK:-2}"
CKDIR="outputs/checkpoints/model_class/cf/lba_finetune"
STEM="cogformer_lba_finetune_s2_A${LBA_FIXED_A}_l${DEC_LAYERS}_h${HEADS}_p${PROJ}_s${SEEDS}_d${MC_DIM}_b256_e${LBA_FT_EPOCHS}_t${STEPS}"
OUTROOT="outputs/data/model_class/cf/lba_trajectory"
LOG="${LOG:-logs/restyle/lba_traj_validate.log}"
mkdir -p "$(dirname "$LOG")"

echo "=== LBA TRAJECTORY VALIDATE start $(date -Is) seed=$SEED ===" | tee -a "$LOG"
for EP in 0 500 1000 1500 2000; do
    CKPT="$CKDIR/${STEM}_ep${EP}.pt"
    OUT="$OUTROOT/ep${EP}"
    if [ ! -f "$CKPT" ]; then echo "MISSING $CKPT" | tee -a "$LOG"; exit 1; fi
    mkdir -p "$OUT"
    echo "=== [ep${EP}] start $(date -Is) ===" | tee -a "$LOG"
    .venv/bin/python -m experiments.model_class.model_class_lba_finetune_validate \
        --lba_fixed_A "$LBA_FIXED_A" \
        --self_attn_padding_mask \
        --max_num_regressors "$MC_MAX_REG" --max_num_categories "$MC_MAX_CAT" \
        --checkpoint "$CKPT" \
        --encoder_num_layers "$ENC_LAYERS" --decoder_num_layers "$DEC_LAYERS" \
        --encoder_num_heads "$HEADS" --decoder_num_heads "$HEADS" \
        --proj_dim "$PROJ" --num_seeds "$SEEDS" --seed_dim "$MC_DIM" \
        --model_embed_dim 8 \
        --batch_size 200 --num_samples 200 --num_obs "$OBS" \
        --samples_chunk_size "$CHUNK" \
        --seed "$SEED" \
        --outdir "$OUT" --pred_dir "$OUT/predictions" >>"$LOG" 2>&1
    echo "=== [ep${EP}] exit=$? $(date -Is) ===" | tee -a "$LOG"
done
echo "=== LBA TRAJECTORY VALIDATE ALL DONE $(date -Is) ===" | tee -a "$LOG"
