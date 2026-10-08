#!/bin/bash
# Full rerun after the ModelClass.collate per_trial_context fix (9394c50, 2026-10-06):
# every model-class checkpoint before it trained with favored_idx = 0 for RDM/LBA.
# New MC_TAG suffix _ptc keeps the enc128 outputs side by side. BF baselines are kept.
#   1. MOVE (never delete) outputs whose completion markers sit at fixed paths, + finals
#   2. e1_train on GPU 0
#   3. two lanes in parallel, one per GPU, each in dependency order
#   4. the three-arm exp3 stages (need both lanes), then finalize
# Launch: setsid nohup bash logs/chain_ptc.sh > logs/chain_ptc.out 2>&1 < /dev/null &
set -uo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2
export PATH="$PWD/.venv/bin:$PATH" VIRTUAL_ENV="$PWD/.venv" NUM_GPU_JOBS=2
source experiments/shared_config.sh
log() { echo "$(date '+%F %T') $*"; }
[[ "$MC_TAG" == *_ptc ]] || { log "ABORT: MC_TAG=$MC_TAG lacks _ptc"; exit 1; }

# ── archive stale fixed-path outputs (mv only; explicit literal paths) ─────────
ARCHIVE=outputs/archive/pre_ptc_20261006
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

# ── e1_train (GPU 0) ───────────────────────────────────────────────────────────
log "e1_train starting (MC_TAG=$MC_TAG)"
CUDA_VISIBLE_DEVICES=0 taskset -c 0-23 bash run_experiments.sh e1_train > logs/chain_ptc_train.out 2>&1
RT=$?
if [[ $RT -ne 0 || ! -f "${MC_CKPT}.done" ]]; then
    log "ABORT: e1_train exited $RT without ${MC_CKPT}.done -- see logs/chain_ptc_train.out"
    exit 1
fi
log "training done"

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
lane_a > logs/chain_ptc_laneA.out 2>&1 & PA=$!
lane_b > logs/chain_ptc_laneB.out 2>&1 & PB=$!
wait $PA; RA=$?; log "lane A exited $RA"
wait $PB; RB=$?; log "lane B exited $RB"
if [[ $RA -ne 0 || $RB -ne 0 ]]; then
    log "ABORT before the exp3 comparison + finalize: a lane failed (see logs/chain_ptc_lane*.out)"
    exit 1
fi

# ── needs both lanes: e1_validate (A) and e3_ft_validate (B) ───────────────────
CUDA_VISIBLE_DEVICES=0 taskset -c 0-23 bash run_experiments.sh \
    e3_original e3_paired_c2st e3_compare \
    e1_finalize e1a_finalize e2_finalize e3_finalize e3_gap_finalize e4_finalize \
    > logs/chain_ptc_final.out 2>&1
RF=$?
log "exp3 comparison + finalize exited $RF"
[[ $RF -eq 0 ]] && log "ALL DONE"
exit $RF
