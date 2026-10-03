#!/bin/bash
# Auto-chain the enc128/alpha0.5/RK4 rerun (2026-10-03) after e1_train finishes.
#   1. wait for the model-class checkpoint's .done sentinel (abort if training dies without it)
#   2. MOVE (never delete) the previous architecture's outputs whose completion markers sit
#      at fixed paths, so their stages re-run instead of being skipped; plus the old finals
#   3. two lanes in parallel, one per GPU, each in dependency order
#   4. the three-arm exp3 stages (need both lanes), then finalize
# Launch: setsid nohup bash logs/chain_enc128.sh > logs/chain_enc128.out 2>&1 < /dev/null &
set -uo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2
export PATH="$PWD/.venv/bin:$PATH" VIRTUAL_ENV="$PWD/.venv" NUM_GPU_JOBS=2
source experiments/shared_config.sh
log() { echo "$(date '+%F %T') $*"; }

TRAIN_PID=${TRAIN_PID:?set TRAIN_PID to the e1_train orchestrator pid}
log "waiting for ${MC_CKPT}.done (training orchestrator pid $TRAIN_PID)"
while [[ ! -f "${MC_CKPT}.done" ]]; do
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        sleep 30   # the sentinel is touched right after the trainer exits
        [[ -f "${MC_CKPT}.done" ]] && break
        log "ABORT: e1_train exited without ${MC_CKPT}.done -- check its stage log under logs/finals/"
        exit 1
    fi
    sleep 300
done
log "training done"

# ── archive stale fixed-path outputs (mv only; explicit literal paths) ─────────
ARCHIVE=outputs/archive/pre_enc128_20261003
mkdir -p "$ARCHIVE"
for p in \
    outputs/data/real_data/ddm \
    outputs/data/real_data/rdm \
    outputs/data/real_data/cdm \
    outputs/data/real_data/lba_stim_cond \
    outputs/figures/real_data/ppc \
    outputs/data/encoder_rep \
    outputs/figures/c2st/ddm \
    outputs/figures/c2st/rdm \
    outputs/figures/c2st/cdm \
    outputs/figures/c2st/lba \
    outputs/finals/experiment-1 \
    outputs/finals/experiment-1a \
    outputs/finals/experiment-2 \
    outputs/finals/experiment-3 \
    outputs/finals/experiment-4
do
    if [[ -e "$p" ]]; then
        mkdir -p "$ARCHIVE/$(dirname "$p")"
        mv -n "$p" "$ARCHIVE/$p" && log "archived $p"
    fi
done

# ── two lanes ──────────────────────────────────────────────────────────────────
# Lane A (GPU 0): exp1, the e1a gap C2ST, exp2 (e2_validate for DDM only).
# Lane B (GPU 1): LBA finetune, exp3 per-arm stages, exp4, LBA gap C2ST, e2_validate RDM+CDM.
# e2_validate's marker is the CDM file, so lane A's DDM part runs with FORCE (scoped to
# that one stage) in case lane B finished CDM first.
lane_a() {
    CUDA_VISIBLE_DEVICES=0 taskset -c 0-23 bash run_experiments.sh \
        e1_validate e1_ensemble e1_benchmark e1_design_eig e1a_c2st \
        e2_apply e2_refit_svfix e2_ppc || return 1
    CUDA_VISIBLE_DEVICES=0 taskset -c 0-23 env FORCE=1 E2_FAMILIES=ddm \
        bash run_experiments.sh e2_validate
}
lane_b() {
    CUDA_VISIBLE_DEVICES=1 taskset -c 24-47 bash run_experiments.sh \
        e3_finetune e3_zeroshot e3_ft_validate e3_gap_c2st \
        e4_probe e4_compare e4_representation \
        e3_ppc_apply e3_ppc e3_realdata_validate || return 1
    CUDA_VISIBLE_DEVICES=1 taskset -c 24-47 env E2_FAMILIES="rdm cdm" \
        bash run_experiments.sh e2_validate
}
log "lane A (GPU 0) and lane B (GPU 1) starting"
lane_a > logs/chain_enc128_laneA.out 2>&1 & PA=$!
lane_b > logs/chain_enc128_laneB.out 2>&1 & PB=$!
wait $PA; RA=$?; log "lane A exited $RA"
wait $PB; RB=$?; log "lane B exited $RB"
if [[ $RA -ne 0 || $RB -ne 0 ]]; then
    log "ABORT before the exp3 comparison + finalize: a lane failed (see logs/chain_enc128_lane*.out)"
    exit 1
fi

# ── needs both lanes: e1_validate (A) and e3_ft_validate (B) ───────────────────
CUDA_VISIBLE_DEVICES=0 taskset -c 0-23 bash run_experiments.sh \
    e3_original e3_paired_c2st e3_compare \
    e1_finalize e1a_finalize e2_finalize e3_finalize e3_gap_finalize e4_finalize \
    > logs/chain_enc128_final.out 2>&1
RF=$?
log "exp3 comparison + finalize exited $RF"
[[ $RF -eq 0 ]] && log "ALL DONE"
exit $RF
