#!/usr/bin/env bash
# 500-epoch, 7-dim-target reruns of BOTH aDDM arms, on the project's pinned
# bayesflow (best-defaults). Sequential: one GPU.
#   arm 1  frozen CogFormer encoder + FlowMatching   (~2h15m, encoder on GPU)
#   arm 2  SetTransformer + FlowMatching baseline    (~2h15m)
# encoder_device cuda: the default is cpu, which extrapolates to 8h36m for the
# same run. Same frozen weights, same forward pass, just not on the CPU.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source experiments/shared_config.sh
LOG=logs/real_data/addm_7d_e500.log
.venv/bin/python -c "import bayesflow;print('bayesflow',bayesflow.__version__)" 2>/dev/null | tail -1 | tee -a "$LOG"

for mode in train validate; do
    echo "=== [frozen] $mode START $(date -Is) (500ep, 7-dim) ===" | tee -a "$LOG"
    .venv/bin/python -m experiments.encoder_rep.addm_bf_apply \
        --mode "$mode" --case intercept_only --first_fixation item0 \
        --num_models 3 --encoder_device cuda --epochs 500 \
        --cogformer_checkpoint "$MC_CKPT" >>"$LOG" 2>&1
    rc=$?; echo "=== [frozen] $mode exit=$rc $(date -Is) ===" | tee -a "$LOG"
    if [ $rc -ne 0 ]; then echo "FROZEN ARM FAILED" | tee -a "$LOG"; exit $rc; fi
done

for mode in train validate; do
    echo "=== [bd_7d_e500] $mode START $(date -Is) (500ep, 7-dim) ===" | tee -a "$LOG"
    .venv/bin/python -m experiments.encoder_rep.addm_bf_baseline \
        --mode "$mode" --case intercept_only --first_fixation item0 \
        --tag bd_7d_e500 --subnet_dropout 0 --epochs 500 >>"$LOG" 2>&1
    rc=$?; echo "=== [bd_7d_e500] $mode exit=$rc $(date -Is) ===" | tee -a "$LOG"
    if [ $rc -ne 0 ]; then echo "BASELINE ARM FAILED" | tee -a "$LOG"; exit $rc; fi
done
echo "7D E500 ALL DONE $(date -Is)" | tee -a "$LOG"
