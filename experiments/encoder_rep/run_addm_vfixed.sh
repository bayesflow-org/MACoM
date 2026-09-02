#!/usr/bin/env bash
# v-FIXED reruns of BOTH aDDM arms. Sequential: one GPU.
#
# WHY: on (rt, choice) alone with constant item values, the data depend on
# (v, v_diff, omega) only through mu = (1+omega)/2 * v_diff and
# Delta = (1-omega)(2v + v_diff) -- three parameters, two functionals, so one
# flat direction, spanned by v. Pinning v to its prior median (V_FIXED = 1.0375
# = shifted_softplus(0.6)) makes (v_diff, omega) <-> (mu, Delta) a bijection and
# leaves a 6-cell target: v_diff, omega, a, tau, s_v, s_tau.
#
# s_v is expected to stay weak -- it is the classic second-order DDM parameter,
# not an aDDM pathology, and no reparameterization rescues it at 500 trials.
#
# Outputs go to *_vfixed trees (addm_vfixed, addm_baseline_vfixed) so the 7-dim
# results of 2026-08-26 stay intact.
#   arm 1  frozen CogFormer encoder + FlowMatching   (~1h50m, encoder on GPU)
#   arm 2  SetTransformer + FlowMatching baseline    (~2h15m)
#
# Validation is EQUALISED across arms at 500 datasets x 500 draws. The 7-dim run
# used 200x500 (frozen) vs 500x100 (baseline), which made small cross-arm gaps
# unreadable against Monte Carlo error. That equalisation is what forced
# --sample_chunk_size: approximator.sample flattens datasets x draws into a single
# batch, and 500x500 peaked at 29 GiB of 31.4 GiB inside FlowMatching's FiLM layer.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source experiments/shared_config.sh

EPOCHS_VF="${EPOCHS_VF:-500}"     # == the 7-dim run, so the two are comparable
VAL_N="${VAL_N:-500}"
VAL_DRAWS="${VAL_DRAWS:-500}"
CHUNK="${CHUNK:-50}"              # datasets per approximator.sample call
BASE_TAG="${BASE_TAG:-vfixed_e500}"
LOG="${LOG:-logs/real_data/addm_vfixed.log}"
mkdir -p "$(dirname "$LOG")"

# Training is the expensive half (~1h50m + ~2h15m); validation is minutes. A
# validation failure must not cost another training run, so each arm's train
# stage is skipped when its checkpoint already exists. Set FORCE_TRAIN=1 to
# retrain regardless -- required whenever the simulator or design config changes,
# since an existing checkpoint says nothing about what it was trained on.
FROZEN_CKPT="outputs/checkpoints/encoder_rep/bf/addm_vfixed/intercept_only/addm_encoder_rep_fm/model.keras"
BASELINE_CKPT="outputs/checkpoints/encoder_rep/bf_baseline/addm_vfixed/intercept_only/item0_${BASE_TAG}/addm_baseline_fm/model.keras"

skip_train() {  # $1 = checkpoint path, $2 = arm label
    if [ -z "${FORCE_TRAIN:-}" ] && [ -f "$1" ]; then
        echo "=== [$2] train SKIPPED (checkpoint present; FORCE_TRAIN=1 to override) $(date -Is) ===" | tee -a "$LOG"
        return 0
    fi
    return 1
}

.venv/bin/python -c "import bayesflow;print('bayesflow',bayesflow.__version__)" 2>/dev/null | tail -1 | tee -a "$LOG"
echo "=== v-FIXED RUN epochs=$EPOCHS_VF val=${VAL_N}x${VAL_DRAWS} $(date -Is) ===" | tee -a "$LOG"

for mode in train validate; do
    [ "$mode" = train ] && skip_train "$FROZEN_CKPT" "frozen/vfixed" && continue
    echo "=== [frozen/vfixed] $mode START $(date -Is) ===" | tee -a "$LOG"
    .venv/bin/python -m experiments.encoder_rep.addm_bf_apply \
        --mode "$mode" --case intercept_only --first_fixation item0 --fix_v \
        --num_models 3 --encoder_device cuda --epochs "$EPOCHS_VF" \
        --val_batch_size "$VAL_N" --num_samples "$VAL_DRAWS" \
        --sample_chunk_size "$CHUNK" \
        --cogformer_checkpoint "$MC_CKPT" >>"$LOG" 2>&1
    rc=$?; echo "=== [frozen/vfixed] $mode exit=$rc $(date -Is) ===" | tee -a "$LOG"
    if [ $rc -ne 0 ]; then echo "FROZEN ARM FAILED" | tee -a "$LOG"; exit $rc; fi
done

# Diagnostics + metrics CSV for the frozen arm (the baseline arm writes its own
# inside validate). Names come from the saved param_mask, so the 6-cell target
# is labelled correctly.
echo "=== [frozen/vfixed] diagnostics START $(date -Is) ===" | tee -a "$LOG"
.venv/bin/python -m experiments.encoder_rep.addm_diagnostics \
    --case intercept_only \
    --data ./outputs/data/encoder_rep/addm_vfixed/intercept_only/addm_encoder_rep_data.npz \
    >>"$LOG" 2>&1
echo "=== [frozen/vfixed] diagnostics exit=$? $(date -Is) ===" | tee -a "$LOG"

for mode in train validate; do
    [ "$mode" = train ] && skip_train "$BASELINE_CKPT" "baseline/vfixed" && continue
    echo "=== [baseline/vfixed] $mode START $(date -Is) ===" | tee -a "$LOG"
    .venv/bin/python -m experiments.encoder_rep.addm_bf_baseline \
        --mode "$mode" --case intercept_only --first_fixation item0 --fix_v \
        --tag "$BASE_TAG" --subnet_dropout 0 --epochs "$EPOCHS_VF" \
        --val_batch_size "$VAL_N" --num_samples "$VAL_DRAWS" \
        --sample_chunk_size "$CHUNK" >>"$LOG" 2>&1
    rc=$?; echo "=== [baseline/vfixed] $mode exit=$rc $(date -Is) ===" | tee -a "$LOG"
    if [ $rc -ne 0 ]; then echo "BASELINE ARM FAILED" | tee -a "$LOG"; exit $rc; fi
done

echo "V-FIXED ALL DONE $(date -Is)" | tee -a "$LOG"
