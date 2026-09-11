#!/usr/bin/env bash
# run_experiments.sh — unified orchestrator for everything that lands in
# outputs/finals/experiment-{1,1a,2,3,4}/.
#
# One entry point, one config source (experiments/shared_config.sh), one log tree.
# Supersedes run_experiment.sh (root, stale: built around the pre-reorg
# model_family/ layout) and experiments/run_pipeline.sh (no finalize stages).
# Those still work standalone; nothing here deletes them.
#
# ── WHAT THIS PRODUCES ────────────────────────────────────────────────────────
# Every stage writes into the source trees that experiments/finalize_experiment_N.py
# reads, and the five `*_finalize` stages then assemble outputs/finals/experiment-N/.
# The finalize stages are idempotent and non-destructive: they copy, carry forward
# manifest rows whose sources are gone, and never delete. Running only the finalize
# group is therefore always safe.
#
# ── USAGE ─────────────────────────────────────────────────────────────────────
#   bash run_experiments.sh                    # every stage, dependency order
#   bash run_experiments.sh finalize           # just rebuild outputs/finals/
#   bash run_experiments.sh exp2               # one experiment, produce + finalize
#   bash run_experiments.sh exp1a              # the expensive gap arm, on its own
#   bash run_experiments.sh e1_validate e1_finalize
#   bash run_experiments.sh 9 12               # by number (see --list)
#   bash run_experiments.sh --list             # print the stage table and exit
#
#   DRY_RUN=1 bash run_experiments.sh          # print commands, run nothing
#   FORCE=1   bash run_experiments.sh exp3     # re-run stages whose outputs exist
#   SKIP_PREFLIGHT=1 ...                       # bypass input checks
#
# ── ENVIRONMENT ───────────────────────────────────────────────────────────────
#   USE_WANDB=1        enable wandb logging on training stages
#   NUM_SIM_WORKERS=N  parallel simulation workers
#   DRY_EPOCHS=N       truncate training stages (smoke test)
#   GPU0, GPU1         CUDA device ids for the two-card stages (default 0 and 1)
#
# ── RESUME ────────────────────────────────────────────────────────────────────
# Each stage declares a completion marker (a file or directory it must have
# produced). If the marker exists the stage is SKIPPED unless FORCE=1. That makes
# the whole script re-runnable after an interruption without repeating GPU-weeks,
# and makes `bash run_experiments.sh` a safe way to ask "what is still missing?"
# when combined with DRY_RUN=1.
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
source experiments/shared_config.sh

# MPLBACKEND=Agg is REQUIRED, not tidiness. On this Wayland/Xwayland session the
# inherited DISPLAY=:0 no longer accepts connections, so matplotlib's backend
# auto-resolution blocks forever on the X unix socket — the process parks in
# unix_wait_for_peer at 0% CPU with no error and no timeout. Every stage plots, so
# without this the pipeline hangs on its first figure. (Verified 2026-09-08.)
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1

PY="${PY:-python}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

# ── ModelClass training scale ─────────────────────────────────────────────────
# Literal 256, not MC_BATCH (64): MC_CKPT hardcodes _b256_ because the production
# run used 256, and MC_TAG must match MC_CKPT's directory component.
MC_TRAIN_BATCH=256
MC_TAG="r4c4_b256_e10000"
MC_CKPT_DIR="./outputs/checkpoints/model_class/cf/${MC_TAG}"

FIG_MC="./outputs/figures/model_class/cf"
DATA_MC="./outputs/data/model_class"
# Canonical three-arm LBA tree, consolidated 2026-08-17. The older paths
# (generalization/model_class_lba/cf/paired_zeroshot, cf/paired_orig,
# lba_finetune/paired_ft) are SYMLINKS into these arms — do not "clean up" by
# deleting the targets. finalize_experiment_3.py reads this tree directly, so the
# LBA stages below pass --outdir explicitly: at their DEFAULT outdirs they would
# write somewhere finalize never looks, and the finals tree would silently go stale.
LBA_PAIRED="${FIG_MC}/paired_lba_20260816"
# Amortization-gap trees (see stage notes).
GAP_PRED_DIR="./outputs/data/model_family_paired"
GAP_FIG_DIR="${FIG_MC}_paired"
BF_DATA_DIR="./outputs/data/model_family"
GAP_SPLIT_ROOT="./outputs/data/_mf_split"

FAMILIES=(ddm rdm cdm)
# bench5. "full" is dropped by model_family_c2st and the BF baselines were never
# trained on "mixed_predictor", so both are absent by construction, not omission.
GAP_CASES=(intercept_only fixed regressed fixed_regressed interaction)

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/finals/${RUN_ID}"
mkdir -p "$LOG_DIR"

WANDB=();  [[ -n "${USE_WANDB:-}"       ]] && WANDB=(--use_wandb)
SIMW=();   [[ -n "${NUM_SIM_WORKERS:-}" ]] && SIMW=(--num_sim_workers "$NUM_SIM_WORKERS")
DRY=();    [[ -n "${DRY_EPOCHS:-}"      ]] && DRY=(--dry_run_epochs "$DRY_EPOCHS")

MASK_ARGS=(--self_attn_padding_mask)
[[ "${SELF_ATTN_PADDING_MASK:-true}" == "true" ]] || MASK_ARGS=(--no-self_attn_padding_mask)

# Architecture block shared by train / validate / ensemble / apply. These MUST
# agree: the padding-mask flag changes the forward pass WITHOUT changing
# state_dict keys, so a mismatch is silent and produces wrong numbers, not an error.
MC_ARCH=(
    "${MASK_ARGS[@]}"
    --encoder_num_layers "$ENC_LAYERS" --decoder_num_layers "$DEC_LAYERS"
    --encoder_num_heads  "$HEADS"      --decoder_num_heads  "$HEADS"
    --proj_dim "$PROJ" --num_seeds "$SEEDS" --seed_dim "$MC_DIM"
    --model_embed_dim 8
)
# A checkpoint's encoder input_embedding is sized by (max_num_regressors,
# max_num_categories) via build_encoder_input_dim (r4c4 -> 72, r5c5 -> 186), and
# the filename does NOT encode r/c. experiments/utils.py defaults both to 5, so an
# unpinned run silently builds an incompatible encoder under the right filename.
# Every stage passes these explicitly.
MC_GEOM=(--max_num_regressors "$MC_MAX_REG" --max_num_categories "$MC_MAX_CAT")

# ── plumbing ──────────────────────────────────────────────────────────────────
C_OK=$'\033[32m'; C_SKIP=$'\033[90m'; C_ERR=$'\033[31m'; C_OFF=$'\033[0m'
[[ -t 1 ]] || { C_OK=""; C_SKIP=""; C_ERR=""; C_OFF=""; }

CURRENT_STAGE=""
declare -a RESULTS=()

run() {                       # run <label> -- <command...>
    local label="$1"; shift; [[ "${1:-}" == "--" ]] && shift
    local log="${LOG_DIR}/${CURRENT_STAGE}.log"
    echo "    · ${label}"
    if [[ -n "${DRY_RUN:-}" ]]; then printf '        '; printf '%q ' "$@"; echo; return 0; fi
    local t0=$SECONDS
    "$@" >>"$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "${C_ERR}    ! FAILED (exit $rc) after $(( (SECONDS-t0)/60 ))m — $log${C_OFF}"
        tail -20 "$log" 2>/dev/null | sed 's/^/      | /'
        return $rc
    fi
    echo "      ok ($(( (SECONDS-t0)/60 ))m)"
}

# done_if <marker> — completion marker for resume. Echo the marker a stage must
# produce; if it exists and FORCE is unset the stage is skipped.
stage_is_done() {
    [[ -n "${FORCE:-}" ]] && return 1
    [[ -e "$1" ]]
}

need() {                      # need <path> <what produces it>
    [[ -e "$1" ]] && return 0
    echo "${C_ERR}    ! missing input: $1${C_OFF}"
    echo "      produced by: $2"
    return 1
}

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — generalising across design space
# ══════════════════════════════════════════════════════════════════════════════
marker_e1_train() { echo "$MC_CKPT"; }
stage_e1_train() {
    run "ModelClass CF train (r${MC_MAX_REG}c${MC_MAX_CAT}, b${MC_TRAIN_BATCH}, e${EPOCHS})" -- \
        "$PY" -m experiments.experiment_1.model_class_cf_train \
            "${MC_ARCH[@]}" "${MC_GEOM[@]}" \
            --checkpoint_dir "$MC_CKPT_DIR" \
            --min_num_obs "$MIN_OBS" --max_num_obs "$MAX_OBS" \
            --train_batch_size "$MC_TRAIN_BATCH" \
            --val_batch_size 50 --val_samples_chunk_size 1 \
            --epochs "$EPOCHS" --steps_per_epoch "$STEPS" \
            --lr 1e-4 --dropout 0.05 --layer_dropout 0.05 \
            --fm_sample_steps 50 --fm_num_samples 100 \
            "${DRY[@]}" "${WANDB[@]}" "${SIMW[@]}"
}

marker_e1_validate() { echo "${FIG_MC}/${MC_TAG}_validate"; }
stage_e1_validate() {
    need "$MC_CKPT" "stage e1_train" || return 1
    # 6 scenarios x 3 families -> per_case/ in finals, and the *_cf_pred.npz that
    # e1_benchmark and finals' predictions/ both consume.
    run "ModelClass CF validate — 6 scenarios x 3 families" -- \
        "$PY" -m experiments.experiment_1.model_class_cf_validate \
            "${MC_ARCH[@]}" "${MC_GEOM[@]}" \
            --checkpoint "$MC_CKPT" \
            --outdir "${FIG_MC}/${MC_TAG}_validate" \
            --pred_dir "${DATA_MC}/${MC_TAG}_validate" \
            --batch_size 200 --num_samples 200 --num_obs "$OBS" \
            --samples_chunk_size 1
}

marker_e1_ensemble() { echo "${FIG_MC}/${MC_TAG}_ensemble/model_class_ensemble_data.npz"; }
stage_e1_ensemble() {
    need "$MC_CKPT" "stage e1_train" || return 1
    # The marker is data.npz, not the directory: finalize_experiment_1 GUARDS on
    # that file, because pre-2026-09 runs wrote figures but no npz and are the
    # stale set this stage overwrites. A directory-only marker would let a stale
    # run masquerade as complete.
    # samples_chunk_size 1: chunk 2 OOMs the RTX 5000 Ada on ensemble sampling.
    run "ModelClass random-design ensemble" -- \
        "$PY" -m experiments.experiment_1.model_class_cf_ensemble \
            "${MC_ARCH[@]}" "${MC_GEOM[@]}" \
            --checkpoint "$MC_CKPT" \
            --outdir "${FIG_MC}/${MC_TAG}_ensemble" \
            --batch_size 200 --num_obs "$OBS" --num_samples 200 \
            --samples_chunk_size 1 --families "DDM,RDM,CDM"
}

marker_e1_benchmark() { echo "${FIG_MC}/${MC_TAG}_ensemble_benchmark"; }
stage_e1_benchmark() {
    need "${DATA_MC}/${MC_TAG}_validate" "stage e1_validate" || return 1
    # --pred_dir pinned to the validate run's own directory. The script's DEFAULT
    # is ./outputs/data/model_class/, a mixed-vintage tree — that default is how a
    # "current" figure gets built from stale predictions with no error at all.
    run "benchmark ensembles (bench5 / bench6 x 3 families)" -- \
        "$PY" -m experiments.experiment_1.model_class_cf_benchmark_ensemble \
            --pred_dir "${DATA_MC}/${MC_TAG}_validate" \
            --outdir "${FIG_MC}/${MC_TAG}_ensemble_benchmark" \
            --max_num_regressors "$MC_MAX_REG" --max_num_categories "$MC_MAX_CAT"
}

marker_e1_design_eig() { echo "./outputs/data/design_opt/ddm/design_eig.csv"; }
stage_e1_design_eig() {
    need "$MC_CKPT" "stage e1_train" || return 1
    run "information gain vs trial count (static design optimisation)" -- \
        "$PY" -m experiments.experiment_1.ddm_design_eig \
            --checkpoint "$MC_CKPT" \
            --outdir "./outputs/data/design_opt/ddm" \
            --max_num_regressors "$MC_MAX_REG" --max_num_categories "$MC_MAX_CAT" \
            --num_obs_grid "200,400,600,800,1000" \
            --batch_size 200 --num_samples 200 --samples_chunk_size 1
}

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1a — amortization gap (split out: 15 BayesFlow trainings)
# ══════════════════════════════════════════════════════════════════════════════
# Its own experiment because it is the only part of the Experiment-1 story that
# cannot be produced from the MC checkpoint alone. Keeping it in exp1 meant a
# cheap re-plot of the per-case figures sat behind a stage nobody wanted to
# trigger. Outputs land in outputs/finals/experiment-1a/.
# The gap is a paired C2ST: per bench5 case, how distinguishable are the posteriors
# of a case-specific BayesFlow baseline from the single amortized ModelClass
# CogFormer? 0.5 = indistinguishable = no gap.
#
# STATE AS OF 2026-09-11: the DDM and CDM gap results already sit in
# outputs/finals/experiment-1/amortization_gap/, but their SOURCES are gone — only
# 1 of 15 BF checkpoints survives and there are zero *_data.npz. RDM has no arm at
# all and never did. So these three stages are a FULL REBUILD (~15 BF trainings),
# not a refresh. finalize_experiment_1 carries the existing DDM/CDM rows forward,
# so skipping this group leaves the current finals tree intact.
marker_e1a_bf() { echo "${BF_DATA_DIR}/rdm_interaction_data.npz"; }
stage_e1a_bf() {
    for fam in "${FAMILIES[@]}"; do
        for case in "${GAP_CASES[@]}"; do
            run "BF baseline train ${fam}/${case}" -- \
                "$PY" -m experiments.experiment_1a.model_family_bf_train \
                    --model "$fam" --case "$case" \
                    --epochs "$MF_BF_EPOCHS" --steps_per_epoch 100 --batch_size 64 \
                    "${DRY[@]}" "${WANDB[@]}" || return 1
            # XLA_PYTHON_CLIENT_PREALLOCATE=false: jax otherwise grabs 75% of the
            # card on a shared box.
            run "BF baseline validate ${fam}/${case}" -- \
                env XLA_PYTHON_CLIENT_PREALLOCATE=false \
                "$PY" -m experiments.experiment_1a.model_family_bf_validate \
                    --model "$fam" --case "$case" \
                    --batch_size "$VAL_BATCH" --num_samples 200 \
                    --skip_posteriors || return 1
        done
    done
}

marker_e1a_paired() { echo "${GAP_PRED_DIR}/rdm_interaction_cf_pred.npz"; }
stage_e1a_paired() {
    need "${BF_DATA_DIR}" "stage e1a_bf" || return 1
    # WHY PAIRED MATTERS. C2ST scores two sets of draws over the SAME simulated
    # datasets, so this re-runs CogFormer validation with --data_dir pointing at
    # the BF npz — loading BF's rts/choices/design_matrices instead of simulating
    # its own. The pre-existing predictions in outputs/data/model_class/ were NOT
    # produced this way, and model_family_c2st's only guard is a shape check on the
    # parameter grid (both arms are 181x7 regardless), so feeding it those files
    # yields a silently-wrong, inflated gap. Writes to its own PRED_DIR.
    #
    # model_class_cf_validate has no --models flag, but in paired mode it skips any
    # case whose npz is absent — so each family gets a view directory holding only
    # its own npz. Filenames are disjoint and seeding is per-(model,case), so
    # splitting changes no result.
    mkdir -p "$GAP_PRED_DIR" "$GAP_FIG_DIR"
    rm -rf "$GAP_SPLIT_ROOT"
    for fam in "${FAMILIES[@]}"; do
        mkdir -p "$GAP_SPLIT_ROOT/$fam"
        for f in "$BF_DATA_DIR/${fam}_"*_data.npz; do
            [[ -e "$f" ]] && ln -sf "$(realpath "$f")" "$GAP_SPLIT_ROOT/$fam/$(basename "$f")"
        done
    done
    for fam in "${FAMILIES[@]}"; do
        run "paired CogFormer validation — ${fam}" -- \
            "$PY" -m experiments.experiment_1a.model_class_cf_validate \
                "${MC_ARCH[@]}" "${MC_GEOM[@]}" \
                --checkpoint "$MC_CKPT" \
                --data_dir "$GAP_SPLIT_ROOT/$fam" \
                --pred_dir "$GAP_PRED_DIR" --outdir "$GAP_FIG_DIR" \
                --batch_size "$VAL_BATCH" --num_samples 200 --num_obs "$OBS" \
                --samples_chunk_size 1 --skip_posteriors || return 1
    done
}

marker_e1a_c2st() { echo "./outputs/figures/c2st/rdm/rdm_c2st_summary.csv"; }
stage_e1a_c2st() {
    need "$GAP_PRED_DIR" "stage e1a_paired" || return 1
    # Only BF_vs_Original is reachable here: Original_vs_Finetuned additionally
    # needs --ft_pred_dir from the LBA-finetuned checkpoint, which is experiment 3's
    # forgetting study (stage e3_paired_c2st), not this one.
    for fam in "${FAMILIES[@]}"; do
        run "C2ST BF_vs_Original — ${fam}" -- \
            "$PY" -m experiments.experiment_1a.model_family_c2st \
                --model "$fam" \
                --bf_data_dir "$BF_DATA_DIR" \
                --mc_pred_dir "$GAP_PRED_DIR" \
                --scenarios BF_vs_Original \
                --max_num_categories "$MC_MAX_CAT" || return 1
    done
}

marker_e1_finalize() { echo ""; }   # always runs: cheap, idempotent, carries rows
stage_e1_finalize() {
    run "assemble outputs/finals/experiment-1" -- \
        "$PY" -m experiments.finalize_experiment_1
}

marker_e1a_finalize() { echo ""; }
stage_e1a_finalize() {
    # Carries forward manifest rows whose sources are gone — for DDM/CDM that is
    # currently the ONLY thing keeping 172 rows attached to 172 surviving files.
    run "assemble outputs/finals/experiment-1a" -- \
        "$PY" -m experiments.finalize_experiment_1a
}

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — real-data posterior predictive checks
# ══════════════════════════════════════════════════════════════════════════════
# MODEL-CLASS ONLY, deliberately. finalize_experiment_2 excludes the *_mf_* figures
# because they come from the separate per-family checkpoints — a different lineage,
# and one that does not exist for RDM at all. Two of the three MF checkpoints named
# in shared_config.sh are absent from disk, so a run.sh-style "apply everything"
# stage here would fail on inputs the finals tree does not even want.
marker_e2_apply() { echo "./outputs/data/real_data/cdm/mc"; }
stage_e2_apply() {
    need "$MC_CKPT" "stage e1_train" || return 1
    need "./data" "the empirical datasets (not produced by this pipeline)" || return 1
    for fam in "${FAMILIES[@]}"; do
        run "real-data model-class apply — ${fam}" -- \
            "$PY" -m "experiments.experiment_2.${fam}_mc_cf_apply" \
                --checkpoint "$MC_CKPT" \
                --encoder_num_layers "$ENC_LAYERS" --decoder_num_layers "$DEC_LAYERS" \
                --encoder_num_heads "$HEADS" --decoder_num_heads "$HEADS" \
                --proj_dim "$PROJ" --num_seeds "$SEEDS" --seed_dim "$MC_DIM" \
                --model_embed_dim 8 \
                --max_num_regressors "$MC_MAX_REG" --max_num_categories "$MC_MAX_CAT" \
                || return 1
    done
}

marker_e2_ppc() { echo "./outputs/figures/real_data/ppc/cdm"; }
stage_e2_ppc() {
    need "./outputs/data/real_data" "stage e2_apply" || return 1
    for fam in "${FAMILIES[@]}"; do
        run "posterior predictive checks — ${fam} (mc)" -- \
            "$PY" -m experiments.experiment_2.ppc \
                --model "$fam" --variant mc \
                --max_num_regressors "$MC_MAX_REG" \
                --max_num_categories "$MC_MAX_CAT" || return 1
    done
}

marker_e2_finalize() { echo ""; }
stage_e2_finalize() {
    run "assemble outputs/finals/experiment-2" -- \
        "$PY" -m experiments.finalize_experiment_2
}

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — LBA zero-shot and finetuning
# ══════════════════════════════════════════════════════════════════════════════
# Every stage pins --outdir into $LBA_PAIRED/<arm>. At their defaults these scripts
# write to generalization/model_class_lba/cf/ and model_class/cf/lba_finetune/,
# which finalize_experiment_3 does not read — the run would appear to succeed while
# the finals tree quietly kept the August figures.
marker_e3_zeroshot() { echo "${LBA_PAIRED}/zeroshot"; }
stage_e3_zeroshot() {
    need "$MC_CKPT" "stage e1_train" || return 1
    run "LBA zero-shot validate (A=${LBA_FIXED_A})" -- \
        "$PY" -m experiments.experiment_3.model_class_lba_cf_validate \
            --lba_fixed_A "$LBA_FIXED_A" "${MC_ARCH[@]}" "${MC_GEOM[@]}" \
            --checkpoint "$MC_CKPT" \
            --outdir "${LBA_PAIRED}/zeroshot" \
            --batch_size 200 --num_samples 200 --num_obs "$OBS" \
            --samples_chunk_size 1
}

marker_e3_finetune() { echo "$LBA_FT_CKPT"; }
stage_e3_finetune() {
    need "$MC_CKPT" "stage e1_train" || return 1
    # fixed_A changes what the intrinsic `a` MEANS (threshold vs start-point range),
    # so every LBA arm — finetune, zero-shot, validate — must use the same value.
    # It is in the checkpoint filename for exactly that reason.
    # Scheme 2 = joint DDM+RDM+CDM+LBA, which is what the forgetting study measures.
    run "LBA finetune (scheme ${FINETUNE_SCHEME:-2}, A=${LBA_FIXED_A})" -- \
        "$PY" -m experiments.experiment_3.model_class_cf_finetune_lba \
            --lba_fixed_A "$LBA_FIXED_A" "${MC_ARCH[@]}" "${MC_GEOM[@]}" \
            --pretrained_checkpoint "$MC_CKPT" \
            --epochs "$LBA_FT_EPOCHS" --steps_per_epoch "$STEPS" \
            --lr 1e-5 --dropout 0.05 --layer_dropout 0.05 \
            --train_batch_size "$LBA_FT_BATCH" \
            --val_batch_size 50 --val_samples_chunk_size 1 \
            --finetune_scheme "${FINETUNE_SCHEME:-2}" \
            "${DRY[@]}" "${WANDB[@]}"
}

marker_e3_ft_validate() { echo "${LBA_PAIRED}/finetuned"; }
stage_e3_ft_validate() {
    need "$LBA_FT_CKPT" "stage e3_finetune" || return 1
    run "LBA finetuned validate — 4 families" -- \
        "$PY" -m experiments.experiment_3.model_class_lba_finetune_validate \
            --lba_fixed_A "$LBA_FIXED_A" "${MC_ARCH[@]}" "${MC_GEOM[@]}" \
            --checkpoint "$LBA_FT_CKPT" \
            --outdir "${LBA_PAIRED}/finetuned" \
            --batch_size 200 --num_samples 200 --num_obs "$OBS" \
            --samples_chunk_size 1
}

marker_e3_original() { echo "${LBA_PAIRED}/original"; }
stage_e3_original() {
    need "$MC_CKPT" "stage e1_train" || return 1
    # The "before" arm of the forgetting study: the ORIGINAL checkpoint scored on
    # the same datasets as the finetuned one, so the paired C2ST has a matched pair.
    run "original-checkpoint arm (forgetting baseline)" -- \
        "$PY" -m experiments.experiment_1.model_class_cf_validate \
            "${MC_ARCH[@]}" "${MC_GEOM[@]}" \
            --checkpoint "$MC_CKPT" \
            --outdir "${LBA_PAIRED}/original" \
            --pred_dir "${LBA_PAIRED}/original/predictions" \
            --batch_size 200 --num_samples 200 --num_obs "$OBS" \
            --samples_chunk_size 1
}

marker_e3_paired_c2st() { echo "${LBA_PAIRED}/c2st/forgetting_c2st_summary.csv"; }
stage_e3_paired_c2st() {
    need "${LBA_PAIRED}/original" "stage e3_original"   || return 1
    need "${LBA_PAIRED}/finetuned" "stage e3_ft_validate" || return 1
    # Did finetuning on LBA move the posteriors for the three families the model was
    # originally trained on? Paired, so no re-simulation.
    run "forgetting study — paired C2ST original vs finetuned" -- \
        "$PY" -m experiments.model_class.model_class_paired_c2st \
            --max_num_categories "$MC_MAX_CAT"
}

marker_e3_trajectory() { echo "${FIG_MC}/lba_trajectory/lba_finetuning_trajectory.pdf"; }
stage_e3_trajectory() {
    # Diagnostics at epochs 0/500/1000/1500/2000 on identical seeded datasets —
    # the path between the arms, not the endpoints. Needs the per-epoch finetune
    # checkpoints (…_ep0 … _ep2000), which e3_finetune writes alongside the final one.
    run "LBA finetuning trajectory report" -- \
        "$PY" -m experiments.model_class.lba_trajectory_report \
            --outdir "${FIG_MC}/lba_trajectory"
}

marker_e3_finalize() { echo ""; }
stage_e3_finalize() {
    run "assemble outputs/finals/experiment-3" -- \
        "$PY" -m experiments.finalize_experiment_3
}

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4 — encoder representation (aDDM)
# ══════════════════════════════════════════════════════════════════════════════
# DELEGATED, on purpose. The five variants differ by non-obvious flag combinations
# (--fix_v / --fix_s_v), per-variant output-tree suffixes and baseline tags, and
# equalised validation budgets. Those are already encoded correctly in the tested
# run_addm_*.sh scripts; re-deriving them inline is how the arms end up mismatched.
# The scripts hardcode `.venv/bin/python` but resolve it relative to the project
# root, which is this script's cwd, so they run correctly from here.
E4_SCRIPTS=(
    "encoder_rep/run_addm_7d_e500.sh:intercept_only_7d (7-dim baseline arm)"
    "encoder_rep/run_addm_vfixed.sh:vfixed (v pinned)"
    "encoder_rep/run_addm_varying.sh:varying_values (trial-varying item values)"
    "encoder_rep/run_addm_svfixed.sh:svfixed + vsvfixed (s_v pinned, and both)"
)
marker_e4_probe() { echo "./outputs/data/encoder_rep/addm_vsvfixed"; }
stage_e4_probe() {
    need "$MC_CKPT" "stage e1_train" || return 1
    for entry in "${E4_SCRIPTS[@]}"; do
        local script="${entry%%:*}" desc="${entry#*:}"
        run "aDDM probe — ${desc}" -- bash "experiments/${script}" || return 1
    done
}

marker_e4_representation() { echo "./outputs/data/encoder_rep/representation/crossfamily"; }
stage_e4_representation() {
    need "$MC_CKPT" "stage e1_train" || return 1
    # REPRESENTATION level, not posterior level: RSA (crossnobis) + CKA against a
    # dimensionality-matched null. The frozen summary is 128-d and the learned one
    # 16-d, so no posterior metric can compare them — this is the only level at
    # which the universal-representation claim is testable.
    run "cross-family representation (RSA / CKA)" -- \
        "$PY" -m experiments.experiment_4.crossfamily_representation \
            --cogformer_checkpoint "$MC_CKPT" --num_models 3 || return 1
    run "aDDM representation — varying_values" -- \
        "$PY" -m experiments.experiment_4.addm_representation \
            --case varying_values --tag varying_e500 \
            --cogformer_checkpoint "$MC_CKPT" --num_models 3 || return 1
    run "aDDM representation — v-fixed" -- \
        "$PY" -m experiments.experiment_4.addm_representation \
            --case intercept_only --fix_v --tag vfixed_e500 \
            --cogformer_checkpoint "$MC_CKPT" --num_models 3 || return 1
    run "aDDM representation — s_v-fixed" -- \
        "$PY" -m experiments.experiment_4.addm_representation \
            --case intercept_only --fix_s_v --tag svfixed_e500 \
            --cogformer_checkpoint "$MC_CKPT" --num_models 3 || return 1
    run "aDDM representation — v + s_v fixed" -- \
        "$PY" -m experiments.experiment_4.addm_representation \
            --case intercept_only --fix_v --fix_s_v --tag vsvfixed_e500 \
            --cogformer_checkpoint "$MC_CKPT" --num_models 3
}

marker_e4_finalize() { echo ""; }
stage_e4_finalize() {
    run "assemble outputs/finals/experiment-4" -- \
        "$PY" -m experiments.finalize_experiment_4
}

# ══════════════════════════════════════════════════════════════════════════════
# stage table
# ══════════════════════════════════════════════════════════════════════════════
ALL_STAGES=(
    e1_train e1_validate e1_ensemble e1_benchmark e1_design_eig e1_finalize
    e1a_bf e1a_paired e1a_c2st e1a_finalize
    e2_apply e2_ppc e2_finalize
    e3_zeroshot e3_finetune e3_ft_validate e3_original e3_paired_c2st
    e3_trajectory e3_finalize
    e4_probe e4_representation e4_finalize
)
GROUP_exp1="e1_train e1_validate e1_ensemble e1_benchmark e1_design_eig e1_finalize"
GROUP_exp1a="e1a_bf e1a_paired e1a_c2st e1a_finalize"
GROUP_exp2="e2_apply e2_ppc e2_finalize"
GROUP_exp3="e3_zeroshot e3_finetune e3_ft_validate e3_original e3_paired_c2st e3_trajectory e3_finalize"
GROUP_exp4="e4_probe e4_representation e4_finalize"
GROUP_finalize="e1_finalize e1a_finalize e2_finalize e3_finalize e4_finalize"

expand_group() {
    case "$1" in
        exp1) echo "$GROUP_exp1" ;;  exp1a) echo "$GROUP_exp1a" ;;
        exp2) echo "$GROUP_exp2" ;;
        exp3) echo "$GROUP_exp3" ;;  exp4) echo "$GROUP_exp4" ;;
        finalize) echo "$GROUP_finalize" ;;
        all)  echo "${ALL_STAGES[*]}" ;;
        *) return 1 ;;
    esac
}

print_table() {
    echo "stage table (number  name  ->  completion marker):"
    local i=1
    for s in "${ALL_STAGES[@]}"; do
        local m; m="$(marker_"$s")"
        local state="   "
        if [[ -z "$m" ]]; then state="${C_SKIP}always${C_OFF}"
        elif [[ -e "$m" ]]; then state="${C_OK}done${C_OFF}"
        else state="${C_ERR}todo${C_OFF}"; fi
        printf '  %2d  %-18s %b  %s\n' "$i" "$s" "$state" "${m:-(idempotent)}"
        i=$((i+1))
    done
    echo
    echo "groups: exp1 exp1a exp2 exp3 exp4 finalize all"
}

# ── selection ─────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--list" || "${1:-}" == "-l" ]]; then print_table; exit 0; fi

SELECTED=()
if [[ $# -eq 0 ]]; then
    SELECTED=("${ALL_STAGES[@]}")
else
    for arg in "$@"; do
        if grp="$(expand_group "$arg")"; then
            for s in $grp; do SELECTED+=("$s"); done
        elif [[ "$arg" =~ ^[0-9]+$ ]] && (( arg >= 1 && arg <= ${#ALL_STAGES[@]} )); then
            SELECTED+=("${ALL_STAGES[$((arg-1))]}")
        else
            found=""
            for s in "${ALL_STAGES[@]}"; do [[ "$s" == "$arg" ]] && { found=1; SELECTED+=("$s"); }; done
            if [[ -z "$found" ]]; then
                echo "unknown stage or group: $arg"; echo; print_table; exit 2
            fi
        fi
    done
fi

# Keep canonical dependency order regardless of the order stages were requested in,
# and drop duplicates from overlapping groups.
ORDERED=()
for s in "${ALL_STAGES[@]}"; do
    for sel in "${SELECTED[@]}"; do
        [[ "$s" == "$sel" ]] && { ORDERED+=("$s"); break; }
    done
done
SELECTED=("${ORDERED[@]}")

echo "════════════════════════════════════════════════════════════════"
echo " CogFormer-2 — finals pipeline   run ${RUN_ID}"
echo " stages : ${SELECTED[*]}"
echo " logs   : ${LOG_DIR}/"
[[ -n "${DRY_RUN:-}" ]] && echo " DRY RUN — no commands will execute"
[[ -n "${FORCE:-}"   ]] && echo " FORCE — completion markers ignored"
echo "════════════════════════════════════════════════════════════════"

PIPE_T0=$SECONDS
FAILED_STAGE=""
for s in "${SELECTED[@]}"; do
    CURRENT_STAGE="$s"
    marker="$(marker_"$s")"
    echo
    if [[ -n "$marker" ]] && stage_is_done "$marker"; then
        echo "${C_SKIP}── $s — skipped (marker exists: $marker)${C_OFF}"
        RESULTS+=("skip  $s")
        continue
    fi
    echo "── $s"
    if "stage_${s}"; then
        RESULTS+=("ok    $s")
    else
        RESULTS+=("FAIL  $s")
        FAILED_STAGE="$s"
        break
    fi
done

echo
echo "════════════════════════════════════════════════════════════════"
for r in "${RESULTS[@]}"; do
    case "$r" in
        ok*)   echo " ${C_OK}${r}${C_OFF}" ;;
        skip*) echo " ${C_SKIP}${r}${C_OFF}" ;;
        *)     echo " ${C_ERR}${r}${C_OFF}" ;;
    esac
done
echo " elapsed: $(( (SECONDS - PIPE_T0) / 60 ))m   logs: ${LOG_DIR}/"
if [[ -n "$FAILED_STAGE" ]]; then
    echo " ${C_ERR}ABORTED at '${FAILED_STAGE}'${C_OFF}"
    echo "════════════════════════════════════════════════════════════════"
    exit 1
fi
echo " ${C_OK}COMPLETE${C_OFF}"
echo "════════════════════════════════════════════════════════════════"
