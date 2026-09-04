#!/usr/bin/env bash
# s_v-FIXED reruns of BOTH aDDM arms, in two variants. Sequential: one GPU.
#
# WHY: under intercept_only with constant item values the data depend on
# (v, v_diff, omega) only through mu = (1+omega)/2 * v_diff and
# Delta = (1-omega)(2v + v_diff) -- two functionals, three parameters, one flat
# direction. run_addm_vfixed.sh closed that by pinning v, which works but gives up
# on v as an inference target: v recovery in the 7-dim and varying-values arms is
# r = 0.006-0.187, i.e. essentially nothing.
#
# The across-trial drift variance is a THIRD functional, s_v^2 * (1 + omega^2).
# Pinning s_v turns it into a constraint on omega, closing the same system from the
# other side and leaving v FREE. That is the question here: does v become
# recoverable when s_v is the parameter given up instead?
#
#   SVFIXED   --fix_s_v            6 cells: v, v_diff, omega, a, tau, s_tau
#   VSVFIXED  --fix_v --fix_s_v    5 cells: v_diff, omega, a, tau, s_tau
#
# The second is the control: it says whether pinning s_v sharpens the parameters
# that were already identified, separately from whether it rescues v.
#
# MEASURED BEFORE RUNNING, so the result can be read against a prior expectation:
# s_v is itself only weakly recovered (r = 0.22-0.34) and its within-posterior
# correlation with both v and omega is ~0.00 in every existing arm. The channel this
# leans on is real but thin, so a null result here is informative, not a failure.
#
# Outputs go to *_svfixed and *_vsvfixed trees; the 7-dim and v-fixed results stay
# untouched.
#   per variant:  arm 1 frozen CogFormer + FlowMatching  (~1h50m)
#                 arm 2 SetTransformer baseline          (~2h15m)
#   two variants => ~8h in total.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source experiments/shared_config.sh

EPOCHS_SV="${EPOCHS_SV:-500}"     # == the 7-dim and v-fixed runs, so all are comparable
VAL_N="${VAL_N:-500}"
VAL_DRAWS="${VAL_DRAWS:-500}"
CHUNK="${CHUNK:-50}"              # 500x500 peaks at 29 GiB without chunking
LOG="${LOG:-logs/real_data/addm_svfixed.log}"
mkdir -p "$(dirname "$LOG")"

# VARIANT : flags : tree-suffix : baseline tag
VARIANTS=(
    "svfixed:--fix_s_v:_svfixed:svfixed_e500"
    "vsvfixed:--fix_v --fix_s_v:_vsvfixed:vsvfixed_e500"
)

skip_train() {  # $1 = checkpoint path, $2 = arm label
    if [ -z "${FORCE_TRAIN:-}" ] && [ -f "$1" ]; then
        echo "=== [$2] train SKIPPED (checkpoint present; FORCE_TRAIN=1 to override) $(date -Is) ===" | tee -a "$LOG"
        return 0
    fi
    return 1
}

.venv/bin/python -c "import bayesflow;print('bayesflow',bayesflow.__version__)" 2>/dev/null | tail -1 | tee -a "$LOG"
echo "=== s_v-FIXED RUN epochs=$EPOCHS_SV val=${VAL_N}x${VAL_DRAWS} $(date -Is) ===" | tee -a "$LOG"

for spec in "${VARIANTS[@]}"; do
    IFS=":" read -r NAME FLAGS SUF TAG <<< "$spec"
    FROZEN_CKPT="outputs/checkpoints/encoder_rep/bf/addm${SUF}/intercept_only/addm_encoder_rep_fm/model.keras"
    BASELINE_CKPT="outputs/checkpoints/encoder_rep/bf_baseline/addm${SUF}/intercept_only/item0_${TAG}/addm_baseline_fm/model.keras"

    echo "=== VARIANT $NAME  flags='$FLAGS'  tree='addm${SUF}' $(date -Is) ===" | tee -a "$LOG"

    for mode in train validate; do
        [ "$mode" = train ] && skip_train "$FROZEN_CKPT" "frozen/$NAME" && continue
        echo "=== [frozen/$NAME] $mode START $(date -Is) ===" | tee -a "$LOG"
        .venv/bin/python -m experiments.encoder_rep.addm_bf_apply \
            --mode "$mode" --case intercept_only --first_fixation item0 $FLAGS \
            --num_models 3 --encoder_device cuda --epochs "$EPOCHS_SV" \
            --val_batch_size "$VAL_N" --num_samples "$VAL_DRAWS" \
            --sample_chunk_size "$CHUNK" \
            --cogformer_checkpoint "$MC_CKPT" >>"$LOG" 2>&1
        rc=$?; echo "=== [frozen/$NAME] $mode exit=$rc $(date -Is) ===" | tee -a "$LOG"
        if [ $rc -ne 0 ]; then echo "FROZEN ARM FAILED ($NAME)" | tee -a "$LOG"; exit $rc; fi
    done

    echo "=== [frozen/$NAME] diagnostics START $(date -Is) ===" | tee -a "$LOG"
    .venv/bin/python -m experiments.encoder_rep.addm_diagnostics \
        --case intercept_only \
        --data "./outputs/data/encoder_rep/addm${SUF}/intercept_only/addm_encoder_rep_data.npz" \
        >>"$LOG" 2>&1
    echo "=== [frozen/$NAME] diagnostics exit=$? $(date -Is) ===" | tee -a "$LOG"

    for mode in train validate; do
        [ "$mode" = train ] && skip_train "$BASELINE_CKPT" "baseline/$NAME" && continue
        echo "=== [baseline/$NAME] $mode START $(date -Is) ===" | tee -a "$LOG"
        .venv/bin/python -m experiments.encoder_rep.addm_bf_baseline \
            --mode "$mode" --case intercept_only --first_fixation item0 $FLAGS \
            --tag "$TAG" --subnet_dropout 0 --epochs "$EPOCHS_SV" \
            --val_batch_size "$VAL_N" --num_samples "$VAL_DRAWS" \
            --sample_chunk_size "$CHUNK" >>"$LOG" 2>&1
        rc=$?; echo "=== [baseline/$NAME] $mode exit=$rc $(date -Is) ===" | tee -a "$LOG"
        if [ $rc -ne 0 ]; then echo "BASELINE ARM FAILED ($NAME)" | tee -a "$LOG"; exit $rc; fi
    done

    echo "=== VARIANT $NAME DONE $(date -Is) ===" | tee -a "$LOG"
done

echo "S_V-FIXED ALL DONE $(date -Is)" | tee -a "$LOG"
