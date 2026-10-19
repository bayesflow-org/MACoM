#!/usr/bin/env bash
# Wait for the gaussian_softmax `main` (pid 4000637) and `gap_train` (pid 4000638)
# orchestrators to exit, then run gap_eval on GPU1 -- unless either reported a FAIL.
set -u
ROOT=/home/jerry/Documents/research/projects/CogFormer-2
STUDY=$ROOT/outputs/gaussian-softmax
cd "$ROOT"

alive() {  # pid still the same run_experiments process (guards against pid reuse)
    tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | grep -q "run_experiments.sh $2"
}

echo "[$(date -Is)] waiting for main (4000637) and gap_train (4000638)"
while alive 4000637 main || alive 4000638 gap_train; do sleep 300; done
echo "[$(date -Is)] both orchestrators exited"

for f in "$STUDY/main.out" "$STUDY/gap_train.out"; do
    if grep -qE 'FAIL' "$f"; then
        echo "[$(date -Is)] FAIL found in $f -- NOT launching gap_eval"; exit 1
    fi
done

echo "[$(date -Is)] launching gap_eval on GPU1"
STUDY_ROOT=outputs/gaussian-softmax \
PY=$ROOT/.venv/bin/python \
CUDA_VISIBLE_DEVICES=1 \
NUMBA_THREADING_LAYER=workqueue \
MPLBACKEND=Agg \
    bash run_experiments.sh gap_eval > "$STUDY/gap_eval.out" 2>&1
rc=$?
echo "[$(date -Is)] gap_eval exited rc=$rc"
