#!/usr/bin/env bash
# gpu0_bf_chain.sh — share the 20 BayesFlow baselines with GPU 0 once `main` is done,
# then run gap_eval when every baseline exists. Written 2026-10-15 for the
# gaussian_softmax study; launched detached, runs IN the study root.
#
# GPU 1's orchestrator (`gap_train`, PID GAP_PID) trains the cases in bf_commands.txt
# order and skips a case whose validated npz already exists. This chain must never train
# a case GPU 1 is on or about to reach, so while GPU 1 is alive it only claims a case from
# the END of the order with at least MARGIN untouched cases between it and GPU 1's
# current case (>= MARGIN x ~2.3 h of GPU-1 work, versus ~2.3 h for the claimed case).
# Once GPU 1's orchestrator has exited (finished, or aborted on a failure), any
# remaining case is fair game. Nothing is ever deleted.
set -uo pipefail

STUDY=/home/jerry/Documents/research/projects/CogFormer-2/outputs/gaussian-softmax
REPO=/home/jerry/Documents/research/projects/CogFormer-2
MAIN_PID=4000637        # bash run_experiments.sh main      (GPU 0)
GAP_PID=4000638         # bash run_experiments.sh gap_train (GPU 1)
MARGIN=2
D="$STUDY/bf_split"
cd "$STUDY"

export PYTHONPATH="$REPO" MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       NCCL_P2P_DISABLE=1 COGFORMER_SIM_BACKEND=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
       NUMBA_THREADING_LAYER=workqueue DESIGN_GENERATOR=gaussian_softmax CUDA_VISIBLE_DEVICES=0
log() { echo "$(date '+%F %T')  $*" >> "$D/chain.log"; }
alive() { kill -0 "$1" 2>/dev/null; }

# Case i (0..19) = lines 2i+1 (train) and 2i+2 (validate) of bf_commands.txt.
mapfile -t CMDS < "$D/bf_commands.txt"
CASES=()
for ((i = 0; i < 20; i++)); do
    m=$(grep -oE -- "--model [a-z]+" <<<"${CMDS[2*i]}" | cut -d' ' -f2)
    c=$(grep -oE -- "--case [a-z_]+" <<<"${CMDS[2*i]}" | cut -d' ' -f2)
    CASES+=("${m}_${c}")
done
done_case()    { [[ -e "./outputs/data/model_family/${CASES[$1]}_data.npz" ]]; }
claimed()      { [[ -d "$D/claims/${CASES[$1]}" ]]; }

log "chain started; waiting for main (PID $MAIN_PID) to exit"
while alive "$MAIN_PID"; do sleep 60; done
log "main exited: $(grep -E 'COMPLETE|ABORTED' "$STUDY/main.out" | tail -1)"

mkdir -p "$D/claims"
while :; do
    # GPU 1's position: first case in order that is neither validated nor ours.
    p=-1
    for ((i = 0; i < 20; i++)); do
        if ! done_case $i && ! claimed $i; then p=$i; break; fi
    done
    (( p < 0 )) && { log "no case left to take"; break; }

    pick=-1
    for ((i = 19; i >= 0; i--)); do
        done_case $i || claimed $i && continue
        if ! alive "$GAP_PID" || (( i - p - 1 >= MARGIN )); then pick=$i; fi
        break
    done
    (( pick < 0 )) && { log "next free case ${CASES[p]} is too close to GPU 1 (margin $MARGIN); stopping"; break; }

    name="${CASES[pick]}"
    mkdir "$D/claims/$name" || { log "claim race on $name"; continue; }
    log "GPU 0 takes $name (GPU 1 at ${CASES[p]}, gap_train alive: $(alive "$GAP_PID" && echo yes || echo no))"
    t0=$SECONDS
    if eval "${CMDS[2*pick]}" > "$D/${name}_train.log" 2>&1 && \
       eval "${CMDS[2*pick+1]}" > "$D/${name}_validate.log" 2>&1; then
        log "$name done ($(( (SECONDS - t0) / 60 )) min)"
    else
        log "$name FAILED -- see $D/${name}_*.log; chain stops"; exit 1
    fi
done

log "waiting for gap_train (PID $GAP_PID) to exit"
while alive "$GAP_PID"; do sleep 60; done
missing=0
for ((i = 0; i < 20; i++)); do done_case $i || { log "missing baseline: ${CASES[i]}"; missing=1; }; done
if (( missing )); then
    # gap_train exited with cases left (it skips e3_bf entirely once GPU 0 has written
    # lba_interaction, its stage marker): rerun this chain to finish them, unconstrained.
    log "baselines missing after gap_train exit; re-entering the claim loop"
    exec bash "$0"
fi
log "all 20 baselines validated"

if grep -q "COMPLETE" "$STUDY/main.out"; then
    log "running gap_eval"
    if (cd "$REPO" && PY="$REPO/.venv/bin/python" USE_WANDB=0 STUDY_ROOT="$STUDY" \
            bash run_experiments.sh gap_eval > "$STUDY/gap_eval.out" 2>&1 < /dev/null); then
        log "gap_eval COMPLETE -- study finished"
    else
        log "gap_eval FAILED -- see $STUDY/gap_eval.out"
    fi
else
    log "main did not complete; gap_eval not run (it needs e1_validate and e3_ft_validate)"
fi
