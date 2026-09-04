#!/usr/bin/env bash
# Post-training analysis for the s_v-fixed aDDM variants: RSA/CKA + paired arm compare.
#
# SPLIT FROM run_addm_svfixed.sh deliberately. That script trains and validates both
# arms, which is the expensive, non-restartable half; these two analyses read the
# resulting checkpoints and are cheap enough to re-run whenever a plot changes. Both
# stages need BOTH arms' checkpoints, so this must run after the training script
# reports ALL DONE.
#
#   representation/  what each arm ENCODES: RSA with crossnobis + CKA against a
#                    dimensionality-matched null, plus the per-parameter RDM
#                    regression. This is the only level at which a 128-d frozen
#                    summary and a 16-d learned one can be compared at all.
#   compare/         what each arm DECODES on SHARED validation datasets: C2ST
#                    against an empirical null, paired log posterior density, CRPS.
#
# Flags MUST match what the checkpoints were trained with, or the simulated data
# distribution diverges from the training one -- the same failure mode the
# fixed_values bug produced. The VARIANTS table below is the single source of truth.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source experiments/shared_config.sh

VAL_N="${VAL_N:-500}"
VAL_DRAWS="${VAL_DRAWS:-500}"
CHUNK="${CHUNK:-50}"
NJOBS="${NJOBS:-12}"
NCOND="${NCOND:-200}"
NREP="${NREP:-4}"
NNULL="${NNULL:-30}"
LOG="${LOG:-logs/real_data/addm_svfixed_analysis.log}"
mkdir -p "$(dirname "$LOG")"

# NAME : flags : baseline tag  (must match run_addm_svfixed.sh)
VARIANTS=(
    "svfixed:--fix_s_v:svfixed_e500"
    "vsvfixed:--fix_v --fix_s_v:vsvfixed_e500"
)

echo "=== s_v-FIXED ANALYSIS START $(date -Is) ===" | tee -a "$LOG"

for spec in "${VARIANTS[@]}"; do
    IFS=":" read -r NAME FLAGS TAG <<< "$spec"
    echo "=== VARIANT $NAME flags='$FLAGS' $(date -Is) ===" | tee -a "$LOG"

    echo "=== [representation/$NAME] START $(date -Is) ===" | tee -a "$LOG"
    .venv/bin/python -m experiments.encoder_rep.addm_representation \
        --case intercept_only --first_fixation item0 $FLAGS \
        --tag "$TAG" --n_conditions "$NCOND" --n_repeats "$NREP" \
        --num_obs 500 --n_null "$NNULL" \
        --cogformer_checkpoint "$MC_CKPT" --encoder_device cuda >>"$LOG" 2>&1
    rc=$?; echo "=== [representation/$NAME] exit=$rc $(date -Is) ===" | tee -a "$LOG"
    [ $rc -ne 0 ] && echo "REPRESENTATION FAILED ($NAME)" | tee -a "$LOG"

    echo "=== [compare/$NAME] START $(date -Is) ===" | tee -a "$LOG"
    .venv/bin/python -m experiments.encoder_rep.addm_arm_compare \
        --case intercept_only --first_fixation item0 $FLAGS \
        --tag "$TAG" --out_tag "intercept_only_${NAME}" \
        --n_val "$VAL_N" --num_samples "$VAL_DRAWS" --sample_chunk_size "$CHUNK" \
        --n_c2st 200 --n_jobs "$NJOBS" \
        --cogformer_checkpoint "$MC_CKPT" --encoder_device cuda >>"$LOG" 2>&1
    rc=$?; echo "=== [compare/$NAME] exit=$rc $(date -Is) ===" | tee -a "$LOG"
    [ $rc -ne 0 ] && echo "COMPARE FAILED ($NAME)" | tee -a "$LOG"
done

echo "S_V-FIXED ANALYSIS ALL DONE $(date -Is)" | tee -a "$LOG"
