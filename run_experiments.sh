#!/usr/bin/env bash
# run_experiments.sh — unified orchestrator for everything that lands in
# outputs/finals/experiment-{1,1a,2,3,4}/.
#
# One entry point, one config source (experiments/shared_config.sh), one log tree.
# The only launcher: the pre-reorg run_experiment.sh, experiments/run_pipeline.sh and the
# per-directory run*.sh scripts were removed on 2026-09-28 (they remain in git history).
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
#   bash run_experiments.sh                    # every stage, dependency order (gap last)
#   bash run_experiments.sh main               # experiments 1 -> 4 WITHOUT the amortization gap
#   bash run_experiments.sh gap_train          # the 20 BayesFlow baselines (needs only e1_data;
#                                              #   run concurrently on another GPU / host)
#   bash run_experiments.sh gap                # gap_train + gap_eval (C2STs, finalize 1a, 3/gap)
#   bash run_experiments.sh finalize           # just rebuild outputs/finals/
#   bash run_experiments.sh exp2               # one experiment, produce + finalize
#   bash run_experiments.sh e1_validate e1_finalize
#   bash run_experiments.sh 9 12               # by number (see --list)
#   bash run_experiments.sh --list             # print the stage table and exit
#
#   DRY_RUN=1 bash run_experiments.sh          # print commands, run nothing
#   FORCE=1   bash run_experiments.sh exp3     # re-run stages whose outputs exist
#   SKIP_PREFLIGHT=1 ...                       # bypass input checks
#
# ── ENVIRONMENT ───────────────────────────────────────────────────────────────
#   USE_WANDB=0|1      wandb tracking of CogFormer training (default TRACK_WANDB=1, shared_config)
#   NUM_SIM_WORKERS=N  parallel simulation workers
#   DRY_EPOCHS=N       truncate training stages (smoke test)
#   BF_FAMILIES="a b"  limit e1a_bf to these families (split baselines across hosts)
#   E2_FAMILIES="a b"  limit e2_validate to these families (split it across GPUs; "ddm"
#                      covers both DDM runs, plain and --fix_s_v). The stage marker is the
#                      CDM file, so a split run is complete only when every subset is done.
#   NPROC=N            train e1_train data-parallel on N GPUs (torchrun). The total batch
#                      stays MC_TRAIN_BATCH (N x MC_TRAIN_BATCH/N per card), so the
#                      simulation budget is unchanged. Pick cards with CUDA_VISIBLE_DEVICES.
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
# STUDY_ROOT is the directory every stage runs IN. All pipeline paths (./outputs, ./logs,
# ./data, wandb's run dir) are cwd-relative, so a study other than the main one -- e.g.
# outputs/gaussian-softmax, set up by experiments/setup_study.sh -- gets its own
# outputs/, finals/ and logs/ without touching the main tree. Its ./study.env sets the
# design generator (read by shared_config.sh). Default: the repo root, the main study.
STUDY_ROOT="$(cd "${STUDY_ROOT:-$PROJECT_ROOT}" && pwd)" || { echo "STUDY_ROOT does not exist" >&2; exit 2; }
cd "$STUDY_ROOT"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
source "${PROJECT_ROOT}/experiments/shared_config.sh" || exit 2
# A non-default generator writing into the main tree would mix two training
# distributions under one outputs/; a study root without study.env would silently run
# the mixture generator. Refuse both.
if [[ "$STUDY_ROOT" == "$PROJECT_ROOT" && "$DESIGN_GENERATOR" != mixture ]]; then
    echo "DESIGN_GENERATOR=$DESIGN_GENERATOR in the main study root; use a STUDY_ROOT (experiments/setup_study.sh)" >&2
    exit 2
fi
if [[ "$STUDY_ROOT" != "$PROJECT_ROOT" && ! -f ./study.env ]]; then
    echo "STUDY_ROOT=$STUDY_ROOT has no study.env; set it up with experiments/setup_study.sh" >&2
    exit 2
fi
# Exported for the record (stage logs, child shells); shared_config.py re-derives it
# from ./study.env itself.
export DESIGN_GENERATOR
GEN=(--design_generator "$DESIGN_GENERATOR")
# Names and tags the e1_train wandb run (wandb-only; checkpoint names carry MC_TAG).
RUN_TAG=(); [[ "$DESIGN_GENERATOR" == mixture ]] || RUN_TAG=(--run_tag "$DESIGN_GENERATOR")
[[ -z "$STRUCTURE_SUFFIX" ]] || RUN_TAG=(--run_tag "${DESIGN_GENERATOR}${STRUCTURE_SUFFIX}")

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
# MC_EPOCHS, MC_TRAIN_BATCH, MC_TAG, MC_CKPT_DIR and MC_CKPT all come from
# shared_config.sh, derived from one another so the tag and filename cannot drift.

FIG_MC="./outputs/figures/model_class/cf"
DATA_MC="./outputs/data/model_class"
# Canonical three-arm LBA tree, consolidated 2026-08-17. The older paths
# (generalization/model_class_lba/cf/paired_zeroshot, cf/paired_orig,
# lba_finetune/paired_ft) are SYMLINKS into these arms — do not "clean up" by
# deleting the targets. finalize_experiment_3.py reads this tree directly, so the
# LBA stages below pass --outdir explicitly: at their DEFAULT outdirs they would
# write somewhere finalize never looks, and the finals tree would silently go stale.
LBA_PAIRED="${FIG_MC}/${MC_TAG}_lba"

# Amortization gap: where BF validation writes its per-case npz (the shared datasets
# plus BF's posterior draws), read by model_family_c2st.
BF_DATA_DIR="./outputs/data/model_family"

FAMILIES=(ddm rdm cdm)
# One seed for every validation arm that is later compared PAIRED (per-case vs
# finetuned, zero-shot vs finetuned). Validators seed each case by
# its (family, scenario) identity, so equal seeds => identical datasets across
# checkpoints. Unseeded, model_class_paired_c2st only WARNS that true_set differs.
VAL_SEED=0
# THE shared benchmark test set (bench5 x DDM/RDM/CDM/LBA), written once by e1_data and
# LOADED by every arm that scores a bench5 case: e1_validate (= Exp 3's original arm),
# e3_zeroshot, e3_ft_validate and the BayesFlow baselines. Loading, not
# re-seeding, is what pairs them: the seed pins parameters but NOT the RTs/choices
# (numba RNG) or the regressor designs. Cases it lacks fall back to simulation.
BENCH_DATA="./outputs/data/benchmark/seed${VAL_SEED}"
# Row layout of predictions scored from BENCH_DATA: the compact bench grid (intercept,
# u_1, u_2, u_1:u_2), i.e. 2 categories. The C2ST scripts use it only to LABEL rows;
# the training value (4) mislabels u_2 and u_1:u_2 as extra categories of u_1.
BENCH_MAX_CAT=2
# bench5. "full" is dropped by model_family_c2st and the BF baselines were never
# trained on "mixed_predictor", so both are absent by construction, not omission.
GAP_CASES=(intercept_only fixed regressed fixed_regressed interaction)

RUN_ID="$(date +%Y%m%d_%H%M%S)${PIPELINE_LABEL:+_${PIPELINE_LABEL}}"
LOG_DIR="logs/finals/${RUN_ID}"
mkdir -p "$LOG_DIR"

USE_WANDB="${USE_WANDB:-$TRACK_WANDB}"
WANDB=();  [[ "$USE_WANDB" == 1 ]] && WANDB=(--use_wandb)
# Fail at launch, not hours into a detached run, when tracking is on but this host
# cannot log to wandb.
if [[ "$USE_WANDB" == 1 && -z "${DRY_RUN:-}" && "${WANDB_MODE:-}" != offline ]]; then
    if ! "$PY" -c "import wandb, sys; sys.exit(0 if wandb.api.api_key else 1)" 2>/dev/null; then
        echo "wandb tracking is on (TRACK_WANDB=1) but this host has no wandb credentials." >&2
        echo "  run 'wandb login' (uv sync --locked --extra tracking), or WANDB_MODE=offline, or USE_WANDB=0" >&2
        exit 1
    fi
fi
SIMW=();   [[ -n "${NUM_SIM_WORKERS:-}" ]] && SIMW=(--num_sim_workers "$NUM_SIM_WORKERS")
DRY=();    [[ -n "${DRY_EPOCHS:-}"      ]] && DRY=(--dry_run_epochs "$DRY_EPOCHS")
# e1_train launcher: plain python, or torchrun (via torch.distributed.run) for NPROC > 1.
MC_LAUNCH=("$PY" -m)
if [[ "${NPROC:-1}" -gt 1 ]]; then
    MC_LAUNCH=("$PY" -m torch.distributed.run --standalone --nproc_per_node "$NPROC" -m)
fi
# One backend for every stage (see shared_config.sh). jax must not preallocate: with GPU
# simulation, torch holds a CUDA context in the same process as every jax stage.
export COGFORMER_SIM_BACKEND="$SIM_BACKEND"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

MASK_ARGS=(--self_attn_padding_mask)
[[ "${SELF_ATTN_PADDING_MASK:-true}" == "true" ]] || MASK_ARGS=(--no-self_attn_padding_mask)
STRUCTURE_ARGS=()
[[ "$ENCODER_STRUCTURE_CONDITIONING" == false ]] || STRUCTURE_ARGS=(--encoder_structure_conditioning)
INTERACTION_ARGS=(--interaction_policy "$INTERACTION_POLICY")
[[ -z "$INTERACTION_PROB" ]] || INTERACTION_ARGS+=(--interaction_prob "$INTERACTION_PROB")

# Architecture block shared by train / validate / ensemble / apply. These MUST
# agree: the padding-mask flag changes the forward pass WITHOUT changing
# state_dict keys, so a mismatch is silent and produces wrong numbers, not an error.
MC_ARCH=(
    "${MASK_ARGS[@]}"
    "${STRUCTURE_ARGS[@]}"
    --encoder_num_layers "$ENC_LAYERS" --decoder_num_layers "$DEC_LAYERS"
    --encoder_num_heads  "$ENC_HEADS"  --decoder_num_heads  "$HEADS"
    --encoder_embed_dim "$ENC_EMBED"
    --proj_dim "$PROJ" --num_seeds "$SEEDS" --seed_dim "$MC_DIM"
    --model_embed_dim 8
)
# The encoder half alone, for scripts whose other architecture defaults already match
# (realdata_validate, ddm_design_eig, experiment 4). Their encoder defaults are the
# pre-2026-10-03 64/8/8, and ENC_HEADS is invisible to a strict load.
MC_ENC=(--encoder_num_layers "$ENC_LAYERS" --encoder_num_heads "$ENC_HEADS" --encoder_embed_dim "$ENC_EMBED" "${STRUCTURE_ARGS[@]}")
# Posterior sampler for every CogFormer scoring stage (see FM_SOLVER in shared_config.sh).
MC_SAMPLER=(--solver "$FM_SOLVER" --num_sample_steps "$FM_SAMPLE_STEPS" --samples_chunk_size "$FM_SAMPLE_CHUNK")
C2ST_ARGS=()
[[ -z "${C2ST_WORKERS:-}" ]] || C2ST_ARGS=(--workers "$C2ST_WORKERS")
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
    # A dry run walks the whole pipeline before anything exists, so every input an
    # upstream stage WOULD produce is missing. Report it and carry on.
    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "    · (dry run) would need: $1  <- $2"; return 0
    fi
    echo "${C_ERR}    ! missing input: $1${C_OFF}"
    echo "      produced by: $2"
    return 1
}

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — generalising across design space
# ══════════════════════════════════════════════════════════════════════════════
marker_e1_data() { echo "${BENCH_DATA}/.done"; }
stage_e1_data() {
    # CPU only, minutes. Never regenerate mid-study: the files ARE the test set, and a
    # re-run yields the same parameters with different datasets, silently unpairing
    # every arm scored on the previous set. The sentinel is what prevents that.
    run "shared benchmark datasets (bench5 x ddm/rdm/cdm/lba, 200 x ${OBS} trials)" -- \
        "$PY" -m experiments.experiment_1.benchmark_data \
            --outdir "$BENCH_DATA" --batch_size 200 --num_obs "$OBS" --seed "$VAL_SEED" "${GEN[@]}" || return 1
    [[ -n "${DRY_RUN:-}" ]] || touch "${BENCH_DATA}/.done"
}

# The trainer rewrites MC_CKPT itself at every validation interval, so the file
# existing says nothing about whether training FINISHED. The marker is a sentinel
# written only after a clean exit, and --resume continues an interrupted run from
# its last save instead of restarting a 58-hour job.
marker_e1_train() { echo "${MC_CKPT}.done"; }
stage_e1_train() {
    run "ModelClass CF train (r${MC_MAX_REG}c${MC_MAX_CAT}, b${MC_TRAIN_BATCH}, e${MC_EPOCHS}, ${NPROC:-1} GPU)" -- \
        "${MC_LAUNCH[@]}" experiments.experiment_1.model_class_cf_train \
            --time_power_law_alpha "$FM_TIME_ALPHA" \
            "${MC_ARCH[@]}" "${MC_GEOM[@]}" "${GEN[@]}" \
            "${INTERACTION_ARGS[@]}" \
            --checkpoint_dir "$MC_CKPT_DIR" \
            --min_num_obs "$MIN_OBS" --max_num_obs "$MAX_OBS" \
            --train_batch_size "$MC_TRAIN_BATCH" \
            --val_batch_size 50 --val_samples_chunk_size 1 \
            --epochs "$MC_EPOCHS" --steps_per_epoch "$STEPS" \
            --lr 1e-4 --dropout 0.05 --layer_dropout 0.05 \
            --fm_sample_steps 50 --fm_num_samples 100 \
            --val_epoch_interval 2500 --early_val_epoch 1000 --resume \
            --fig_dir "${FIG_MC}/${MC_TAG}_train" \
            "${DRY[@]}" "${WANDB[@]}" "${RUN_TAG[@]}" "${SIMW[@]}" || return 1
    # A DRY_EPOCHS smoke test writes under the real filename; never mark it complete.
    [[ -n "${DRY_RUN:-}${DRY_EPOCHS:-}" ]] || touch "${MC_CKPT}.done"
}

marker_e1_validate() { echo "${FIG_MC}/${MC_TAG}_validate"; }
stage_e1_validate() {
    need "${MC_CKPT}.done" "stage e1_train" || return 1
    need "${BENCH_DATA}/.done" "stage e1_data" || return 1
    # 6 scenarios x 3 families -> per_case/ in finals, and the *_cf_pred.npz that
    # e1_benchmark and finals' predictions/ both consume. Seeded with VAL_SEED, so this
    # is ALSO experiment 3's "original" arm (see e3_original), and --skip_posteriors
    # because finalize's allowlist drops the 180 pairplots anyway.
    run "ModelClass CF validate — 6 scenarios x 3 families" -- \
        "$PY" -m experiments.experiment_1.model_class_cf_validate \
            "${MC_SAMPLER[@]}" \
            "${MC_ARCH[@]}" "${MC_GEOM[@]}" "${GEN[@]}" \
            --checkpoint "$MC_CKPT" \
            --outdir "${FIG_MC}/${MC_TAG}_validate" \
            --pred_dir "${DATA_MC}/${MC_TAG}_validate" \
            --batch_size 200 --num_samples 200 --num_obs "$OBS" \
            --seed "$VAL_SEED" --skip_posteriors \
            --data_dir "$BENCH_DATA"
}

marker_e1_ensemble() { echo "${FIG_MC}/${MC_TAG}_ensemble/model_class_ensemble_data.npz"; }
stage_e1_ensemble() {
    need "${MC_CKPT}.done" "stage e1_train" || return 1
    # The marker is data.npz, not the directory: finalize_experiment_1 GUARDS on
    # that file, because pre-2026-09 runs wrote figures but no npz and are the
    # stale set this stage overwrites. A directory-only marker would let a stale
    # run masquerade as complete.
    run "ModelClass random-design ensemble" -- \
        "$PY" -m experiments.experiment_1.model_class_cf_ensemble \
            "${MC_SAMPLER[@]}" \
            "${MC_ARCH[@]}" "${MC_GEOM[@]}" "${GEN[@]}" \
            "${INTERACTION_ARGS[@]}" \
            --checkpoint "$MC_CKPT" \
            --outdir "${FIG_MC}/${MC_TAG}_ensemble" \
            --batch_size 200 --num_obs "$OBS" --num_samples 200 \
            --families "DDM,RDM,CDM"
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

# Scoped to MC_TAG: an untagged directory lets a result from any other checkpoint
# satisfy the marker and be copied into finals as this run's.
DESIGN_OPT_DIR="./outputs/data/design_opt/${MC_TAG}/ddm"
marker_e1_design_eig() { echo "${DESIGN_OPT_DIR}/design_eig.csv"; }
stage_e1_design_eig() {
    need "${MC_CKPT}.done" "stage e1_train" || return 1
    run "information gain vs trial count (static design optimisation)" -- \
        "$PY" -m experiments.experiment_1.ddm_design_eig \
            "${MC_ENC[@]}" "${MC_SAMPLER[@]}" "${GEN[@]}" \
            --checkpoint "$MC_CKPT" \
            --outdir "$DESIGN_OPT_DIR" \
            --max_num_regressors "$MC_MAX_REG" --max_num_categories "$MC_MAX_CAT" \
            --num_obs_grid "200,400,600,800,1000" \
            --batch_size 200 --num_samples 200
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
#
# PAIRING (2026-09-27): both arms score the shared benchmark files from e1_data. The
# CogFormer arm IS e1_validate; the BayesFlow arm loads the same files via --data_dir.
# The BF simulator reads each case's design options from val_scenarios, so the
# baselines train on the distribution they are tested on.
marker_e1a_bf() { echo "${BF_DATA_DIR}/rdm_interaction_data.npz"; }
stage_e1a_bf() {
    # BF_FAMILIES="ddm cdm" restricts this stage to a subset, to split the baselines
    # across hosts; the per-case skip below makes the split resumable either way.
    local fams=(${BF_FAMILIES:-${FAMILIES[*]}})
    for fam in "${fams[@]}"; do
        [[ " ${FAMILIES[*]} " == *" $fam "* ]] || { echo "unknown BF family '$fam' (expected: ${FAMILIES[*]})" >&2; return 1; }
    done
    for fam in "${fams[@]}"; do
        for case in "${GAP_CASES[@]}"; do
            # Per-case resume: the stage marker only names the LAST case, so without
            # this an interruption at case 14 would retrain all 15 (~8 h each at b128).
            # The npz is written by validate, i.e. only after that case's training
            # finished, and it is what e1a_c2st pairs against.
            if [[ -e "${BF_DATA_DIR}/${fam}_${case}_data.npz" && -z "${FORCE:-}" ]]; then
                echo "    · ${fam}/${case}: validated npz exists, skipped"; continue
            fi
            if [[ "${BF_REUSE_CHECKPOINTS:-0}" == 1 ]]; then
                need "./outputs/checkpoints/model_family/bf/${fam}/${fam}_families_bf_${case}/model.keras" "copied compatible Gaussian-softmax BF checkpoint" || return 1
                echo "    · ${fam}/${case}: reusing BF weights; evaluating this study's datasets"
            else
                run "BF baseline train ${fam}/${case}" -- \
                "$PY" -m experiments.experiment_1a.model_family_bf_train \
                    --model "$fam" --case "$case" \
                    --epochs "$MF_BF_EPOCHS" --steps_per_epoch "$STEPS" --batch_size "$MF_BF_BATCH" "${GEN[@]}" \
                    "${DRY[@]}" || return 1
            fi
            # XLA_PYTHON_CLIENT_PREALLOCATE=false: jax otherwise grabs 75% of the
            # card on a shared box.
            need "${BENCH_DATA}/.done" "stage e1_data" || return 1
            run "BF baseline validate ${fam}/${case}" -- \
                env XLA_PYTHON_CLIENT_PREALLOCATE=false \
                "$PY" -m experiments.experiment_1a.model_family_bf_validate \
                    --model "$fam" --case "$case" \
                    --num_samples 200 --data_dir "$BENCH_DATA" "${GEN[@]}" \
                    --skip_posteriors || return 1
        done
    done
}

marker_e1a_c2st() { echo "./outputs/figures/c2st/rdm/rdm_c2st_summary.csv"; }
stage_e1a_c2st() {
    # The CogFormer side is e1_validate's predictions: it scored the SAME shared
    # benchmark files as the baselines, so no separate paired run is needed.
    need "${DATA_MC}/${MC_TAG}_validate" "stage e1_validate" || return 1
    # Only BF_vs_Original is reachable here: Original_vs_Finetuned additionally
    # needs --ft_pred_dir from the LBA-finetuned checkpoint, which is experiment 3's
    # forgetting study (stage e3_paired_c2st), not this one.
    for fam in "${FAMILIES[@]}"; do
        run "C2ST BF_vs_Original — ${fam}" -- \
            "$PY" -m experiments.experiment_1a.model_family_c2st \
                --model "$fam" \
                --bf_data_dir "$BF_DATA_DIR" \
                --mc_pred_dir "${DATA_MC}/${MC_TAG}_validate" \
                --scenarios BF_vs_Original \
                "${C2ST_ARGS[@]}" \
                --max_num_categories "$BENCH_MAX_CAT" || return 1
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
    need "${MC_CKPT}.done" "stage e1_train" || return 1
    need "./data" "the empirical datasets (not produced by this pipeline)" || return 1
    for fam in "${FAMILIES[@]}"; do
        run "real-data model-class apply — ${fam}" -- \
            "$PY" -m "experiments.experiment_2.${fam}_mc_cf_apply" \
                "${MC_SAMPLER[@]}" \
                --checkpoint "$MC_CKPT" \
                --encoder_num_layers "$ENC_LAYERS" --decoder_num_layers "$DEC_LAYERS" \
                --encoder_num_heads "$ENC_HEADS" --decoder_num_heads "$HEADS" \
                --encoder_embed_dim "$ENC_EMBED" \
                "${STRUCTURE_ARGS[@]}" \
                --proj_dim "$PROJ" --num_seeds "$SEEDS" --seed_dim "$MC_DIM" \
                --model_embed_dim 8 \
                --max_num_regressors "$MC_MAX_REG" --max_num_categories "$MC_MAX_CAT" \
                || return 1
    done
}

# Marker is the LAST variant produced, not ppc/cdm: the canonical loop and the
# constrained variants are one stage, so a marker satisfied halfway through would
# report the stage done with the variants missing.
marker_e2_refit_svfix() { echo "./outputs/data/real_data/ddm/mc_sv_fixed/ddm_mc_posteriors.npz"; }
stage_e2_refit_svfix() {
    need "${MC_CKPT}.done" "stage e1_train" || return 1
    need "./data/data_application_ddm.csv" "the empirical DDM dataset (not produced here)" || return 1
    # A genuine constrained FIT, not a simulation-time override: --fix_s_v drops s_v
    # from the design so it is never an estimation target, and every other parameter is
    # re-estimated under that constraint. This is what matches the original DDM
    # specification (Stump et al., 2026); ppc.py's --fix_natural does NOT, because it
    # leaves the other parameters carrying the compensation they acquired while s_v was
    # free. Writes to .../ddm/mc_sv_fixed/ rather than overwriting the canonical fit.
    run "DDM refit with s_v fixed at 0 (original specification)" -- \
        "$PY" -m experiments.experiment_2.ddm_mc_cf_apply \
            "${MC_SAMPLER[@]}" \
            --fix_s_v \
            --checkpoint "$MC_CKPT" \
            --encoder_num_layers "$ENC_LAYERS" --decoder_num_layers "$DEC_LAYERS" \
            --encoder_num_heads "$ENC_HEADS" --decoder_num_heads "$HEADS" \
            --encoder_embed_dim "$ENC_EMBED" \
            "${STRUCTURE_ARGS[@]}" \
            --proj_dim "$PROJ" --num_seeds "$SEEDS" --seed_dim "$MC_DIM" \
            --model_embed_dim 8 \
            --max_num_regressors "$MC_MAX_REG" --max_num_categories "$MC_MAX_CAT"
}

marker_e2_ppc() { echo "./outputs/figures/real_data/ppc/combined_ppc.pdf"; }
stage_e2_ppc() {
    need "./outputs/data/real_data" "stage e2_apply" || return 1
    for fam in "${FAMILIES[@]}"; do
        run "posterior predictive checks — ${fam} (mc)" -- \
            "$PY" -m experiments.experiment_2.ppc \
                --model "$fam" --variant mc \
                --max_num_regressors "$MC_MAX_REG" \
                --max_num_categories "$MC_MAX_CAT" || return 1
    done
    # Constrained variants: same posterior draws, something pinned during the forward
    # simulation. finalize_experiment_2 files these under <family>/<variant>/ so the
    # family root keeps meaning "the model's own fit, simulated as-is".
    run "PPC — cdm, s_v pinned to 0" -- \
        "$PY" -m experiments.experiment_2.ppc \
            --model cdm --variant mc \
            --fix_natural "s_v=0.0" \
            --outdir ./outputs/figures/real_data/ppc/cdm_sv0 \
            --max_num_regressors "$MC_MAX_REG" \
            --max_num_categories "$MC_MAX_CAT" || return 1
    # Matches the ORIGINAL DDM specification: Stump et al. (2026) has no across-trial
    # drift variability, but this checkpoint estimates s_v intercept-only. Last in the
    # stage, and the stage marker, so a partial run cannot report itself complete.
    run "PPC — ddm, s_v pinned at simulation only" -- \
        "$PY" -m experiments.experiment_2.ppc \
            --model ddm --variant mc \
            --fix_natural "s_v=0.0" \
            --outdir ./outputs/figures/real_data/ppc/ddm_sv0 \
            --max_num_regressors "$MC_MAX_REG" \
            --max_num_categories "$MC_MAX_CAT" || return 1
    # The REFIT's PPC. --fix_natural is REQUIRED here even though s_v was fixed at
    # fitting time: a parameter absent from the design has its coefficients masked to
    # zero, and ppc.py still applies the link -- shifted_softplus(0) = 0.693, not 0.
    # Without it this would silently simulate s_v at 0.693.
    if [[ -f ./outputs/data/real_data/ddm/mc_sv_fixed/ddm_mc_posteriors.npz ]]; then
        run "PPC — ddm refit, s_v fixed at fitting (original specification)" -- \
            "$PY" -m experiments.experiment_2.ppc \
                --model ddm --variant mc \
                --posteriors ./outputs/data/real_data/ddm/mc_sv_fixed/ddm_mc_posteriors.npz \
                --fix_natural "s_v=0.0" \
                --outdir ./outputs/figures/real_data/ppc/ddm_sv_fixed \
                --max_num_regressors "$MC_MAX_REG" \
                --max_num_categories "$MC_MAX_CAT"
    else
        echo "    · skipped: ddm s_v-fixed refit absent (run stage e2_refit_svfix)"
    fi
    # All three families in one figure, from the SAME cached draws the per-family runs
    # just simulated (no re-simulation). Last, and the stage marker, so a partial PPC
    # run cannot report itself complete.
    run "PPC — combined figure, all families" -- \
        "$PY" -m experiments.experiment_2.combined_all \
            --variant mc \
            --max_num_regressors "$MC_MAX_REG" --max_num_categories "$MC_MAX_CAT"
}

# Simulation-based validation on each family's EMPIRICAL design: every dataset borrows
# a real subject's regressor values, trial count and per-trial context (RDM: 3
# alternatives with the subject's ink colours), with the parameter design the real-data
# fit used. Shows how well the actual experiment identifies each reported coefficient.
# DDM twice: the canonical design and the s_v-fixed design of the refit.
REALDATA_FIG="${FIG_MC}/${MC_TAG}_realdata"
REALDATA_PRED="${DATA_MC}/${MC_TAG}_realdata"
marker_e2_validate() { echo "${REALDATA_PRED}/cdm_realdata_cf_pred.npz"; }
stage_e2_validate() {
    need "${MC_CKPT}.done" "stage e1_train" || return 1
    need "./data" "the empirical datasets (not produced by this pipeline)" || return 1
    local spec fam flag
    # E2_FAMILIES="ddm" / "rdm cdm" restricts the stage to a subset, to split its ~12 h
    # across GPUs (each subset in its own orchestrator run, pinned with CUDA_VISIBLE_DEVICES).
    local fams=(${E2_FAMILIES:-${FAMILIES[*]}})
    for fam in "${fams[@]}"; do
        [[ " ${FAMILIES[*]} " == *" $fam "* ]] || { echo "unknown E2 family '$fam' (expected: ${FAMILIES[*]})" >&2; return 1; }
    done
    for spec in "ddm:" "ddm:--fix_s_v" "rdm:" "cdm:"; do
        fam="${spec%%:*}"; flag="${spec#*:}"
        [[ " ${fams[*]} " == *" $fam "* ]] || continue
        run "real-data-design validation — ${fam} ${flag}" -- \
            "$PY" -m experiments.experiment_2.realdata_validate \
                "${MC_ENC[@]}" "${MC_SAMPLER[@]}" \
                --model "$fam" $flag \
                --checkpoint "$MC_CKPT" \
                --outdir "$REALDATA_FIG" --pred_dir "$REALDATA_PRED" \
                --num_datasets 200 --num_samples 200 \
                --seed "$VAL_SEED" \
                --max_num_regressors "$MC_MAX_REG" --max_num_categories "$MC_MAX_CAT" || return 1
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
    need "${MC_CKPT}.done" "stage e1_train" || return 1
    need "${BENCH_DATA}/.done" "stage e1_data" || return 1
    # LBA bench5 from the shared files, like the finetuned and BF arms, so the
    # zero-shot -> finetuned change is measured on identical datasets.
    run "LBA zero-shot validate (A=${LBA_FIXED_A})" -- \
        "$PY" -m experiments.experiment_3.model_class_lba_cf_validate \
            "${MC_SAMPLER[@]}" \
            --lba_fixed_A "$LBA_FIXED_A" "${MC_ARCH[@]}" "${MC_GEOM[@]}" "${GEN[@]}" \
            --checkpoint "$MC_CKPT" \
            --outdir "${LBA_PAIRED}/zeroshot" \
            --batch_size 200 --num_samples 200 --num_obs "$OBS" \
            --seed "$VAL_SEED" --data_dir "$BENCH_DATA"
}

# Same sentinel logic as e1_train: the finetune overwrites LBA_FT_CKPT every
# --checkpoint_interval epochs. It has no --resume, so an interrupted finetune
# restarts from epoch 0.
marker_e3_finetune() { echo "${LBA_FT_CKPT}.done"; }
stage_e3_finetune() {
    need "${MC_CKPT}.done" "stage e1_train" || return 1
    # fixed_A changes what the intrinsic `a` MEANS (threshold vs start-point range),
    # so every LBA arm — finetune, zero-shot, validate — must use the same value.
    # It is in the checkpoint filename for exactly that reason.
    # Scheme 2 = joint DDM+RDM+CDM+LBA, which is what the forgetting study measures.
    run "LBA finetune (scheme ${FINETUNE_SCHEME:-2}, A=${LBA_FIXED_A})" -- \
        "$PY" -m experiments.experiment_3.model_class_cf_finetune_lba \
            --time_power_law_alpha "$FM_TIME_ALPHA" --checkpoint_dir "$(dirname "$LBA_FT_CKPT")" \
            --lba_fixed_A "$LBA_FIXED_A" "${MC_ARCH[@]}" "${MC_GEOM[@]}" "${GEN[@]}" \
            "${INTERACTION_ARGS[@]}" \
            --pretrained_checkpoint "$MC_CKPT" \
            --epochs "$LBA_FT_EPOCHS" --steps_per_epoch "$STEPS" \
            --lr 1e-5 --dropout 0.05 --layer_dropout 0.05 \
            --train_batch_size "$LBA_FT_BATCH" \
            --val_batch_size 50 --val_samples_chunk_size 1 \
            --finetune_scheme "${FINETUNE_SCHEME:-2}" \
            --seed "$VAL_SEED" \
            "${DRY[@]}" "${WANDB[@]}" || return 1
    [[ -n "${DRY_RUN:-}${DRY_EPOCHS:-}" ]] || touch "${LBA_FT_CKPT}.done"
}

marker_e3_ft_validate() { echo "${LBA_PAIRED}/finetuned"; }
stage_e3_ft_validate() {
    need "${LBA_FT_CKPT}.done" "stage e3_finetune" || return 1
    run "LBA finetuned validate — 4 families" -- \
        "$PY" -m experiments.experiment_3.model_class_lba_finetune_validate \
            "${MC_SAMPLER[@]}" \
            --lba_fixed_A "$LBA_FIXED_A" "${MC_ARCH[@]}" "${MC_GEOM[@]}" "${GEN[@]}" \
            --checkpoint "$LBA_FT_CKPT" \
            --outdir "${LBA_PAIRED}/finetuned" \
            --pred_dir "${LBA_PAIRED}/finetuned/predictions" \
            --batch_size 200 --num_samples 200 --num_obs "$OBS" \
            --seed "$VAL_SEED" --data_dir "$BENCH_DATA"
}

marker_e3_original() { echo "${LBA_PAIRED}/original"; }
stage_e3_original() {
    # The "before" arm of the forgetting study: the ORIGINAL checkpoint on the same
    # seeded datasets as the finetuned one. That is exactly e1_validate -- same script,
    # checkpoint, scenarios, arguments and VAL_SEED -- so it is LINKED, not re-run.
    # Per-family folders point at e1_validate's figures and predictions/ at its npz,
    # giving model_class_paired_c2st and finalize_experiment_3 the layout they read.
    need "${FIG_MC}/${MC_TAG}_validate" "stage e1_validate" || return 1
    need "${DATA_MC}/${MC_TAG}_validate" "stage e1_validate" || return 1
    [[ -n "${DRY_RUN:-}" ]] && { echo "    · (dry run) would link original/ -> e1_validate"; return 0; }
    local fam out="${LBA_PAIRED}/original"
    mkdir -p "$out"
    for fam in "${FAMILIES[@]}"; do
        ln -sfn "$(realpath "${FIG_MC}/${MC_TAG}_validate/${fam}")" "${out}/${fam}"
    done
    ln -sfn "$(realpath "${DATA_MC}/${MC_TAG}_validate")" "${out}/predictions"
    echo "    · original/ linked to e1_validate (${#FAMILIES[@]} families + predictions)"
}

# ZERO-SHOT -> FINETUNED, and ORIGINAL -> FINETUNED, as paired differences on bench5
# (every arm scored the shared files, which model_class_lba_compare asserts):
#   Δr heatmaps per family and case, and a per-parameter metric-delta table/figure;
#   the calibration plane: each LBA cell's move in (contraction, calibration error).
# CPU only, from saved predictions and metrics.
marker_e3_compare() { echo "${LBA_PAIRED}/compare/delta_metrics.csv"; }
stage_e3_compare() {
    need "${LBA_PAIRED}/zeroshot" "stage e3_zeroshot" || return 1
    need "${LBA_PAIRED}/finetuned" "stage e3_ft_validate" || return 1
    need "${DATA_MC}/${MC_TAG}_validate" "stage e1_validate" || return 1
    run "zero-shot / original vs finetuned — paired deltas" -- \
        "$PY" -m experiments.experiment_3.model_class_lba_compare \
            --zero_shot_dir "${LBA_PAIRED}/zeroshot/A${LBA_FIXED_A}" \
            --finetuned_dir "${LBA_PAIRED}/finetuned" \
            --base_fig_dir "${FIG_MC}/${MC_TAG}_validate" \
            --base_pred_dir "${DATA_MC}/${MC_TAG}_validate" \
            --outdir "${LBA_PAIRED}/compare" \
            --scenarios "${GAP_CASES[@]}" --max_num_categories "$BENCH_MAX_CAT" || return 1
    run "LBA calibration plane (zero-shot -> finetuned)" -- \
        "$PY" -m experiments.experiment_3.calibration_plane \
            --paired_root "$LBA_PAIRED" --scenarios "${GAP_CASES[@]}" \
            --out "${LBA_PAIRED}/compare/lba_calibration_plane.pdf" \
            --csv_out "${LBA_PAIRED}/compare/lba_calibration_plane.csv" || return 1
    # Main-text figure: zero-shot and finetuned on one grid, interaction design.
    run "LBA recovery overlay (zero-shot + finetuned, interaction)" -- \
        "$PY" -m experiments.experiment_3.overlay_recovery \
            --zero_shot_dir "${LBA_PAIRED}/zeroshot/A${LBA_FIXED_A}" \
            --finetuned_dir "${LBA_PAIRED}/finetuned" \
            --outdir "${LBA_PAIRED}/compare" --scenario interaction \
            --max_num_regressors "$MC_MAX_REG"
}

# LBA AMORTIZATION GAP -- the held-out family's counterpart of experiment 1a. LBA has
# no original-checkpoint arm (it was not in the model class), so its CogFormer arm is
# the FINETUNED checkpoint. Same budget per case as the 1a baselines, same shared
# benchmark files (the LBA cases of e1_data), same generator check (verified
# 2026-09-27: mean RT p >= 0.23 in all 5 cases, favored_idx reaches the BF input).
marker_e3_bf() { echo "${BF_DATA_DIR}/lba_interaction_data.npz"; }
stage_e3_bf() {
    need "${BENCH_DATA}/.done" "stage e1_data" || return 1
    local case
    for case in "${GAP_CASES[@]}"; do
        if [[ -e "${BF_DATA_DIR}/lba_${case}_data.npz" && -z "${FORCE:-}" ]]; then
            echo "    · lba/${case}: validated npz exists, skipped"; continue
        fi
        if [[ "${BF_REUSE_CHECKPOINTS:-0}" == 1 ]]; then
            need "./outputs/checkpoints/model_family/bf/lba/lba_families_bf_${case}/model.keras" "copied compatible Gaussian-softmax BF checkpoint" || return 1
            echo "    · lba/${case}: reusing BF weights; evaluating this study's datasets"
        else
            run "BF baseline train lba/${case}" -- \
            "$PY" -m experiments.experiment_1a.model_family_bf_train \
                --model lba --case "$case" \
                --epochs "$MF_BF_EPOCHS" --steps_per_epoch "$STEPS" --batch_size "$MF_BF_BATCH" "${GEN[@]}" \
                "${DRY[@]}" || return 1
        fi
        run "BF baseline validate lba/${case}" -- \
            env XLA_PYTHON_CLIENT_PREALLOCATE=false \
            "$PY" -m experiments.experiment_1a.model_family_bf_validate \
                --model lba --case "$case" \
                --num_samples 200 --data_dir "$BENCH_DATA" "${GEN[@]}" \
                --skip_posteriors || return 1
    done
}

marker_e3_gap_c2st() { echo "./outputs/figures/c2st/lba/lba_c2st_summary.csv"; }
stage_e3_gap_c2st() {
    need "${BF_DATA_DIR}/lba_interaction_data.npz" "stage e3_bf" || return 1
    need "${LBA_PAIRED}/finetuned/predictions" "stage e3_ft_validate" || return 1
    run "C2ST BF_vs_Finetuned — lba" -- \
        "$PY" -m experiments.experiment_1a.model_family_c2st \
            --model lba \
            --bf_data_dir "$BF_DATA_DIR" \
            --ft_pred_dir "${LBA_PAIRED}/finetuned/predictions" \
            --scenarios BF_vs_Finetuned \
            "${C2ST_ARGS[@]}" \
            --max_num_categories "$BENCH_MAX_CAT"
}

marker_e3_paired_c2st() { echo "${LBA_PAIRED}/c2st/forgetting_c2st_summary.csv"; }
stage_e3_paired_c2st() {
    need "${LBA_PAIRED}/original" "stage e3_original"   || return 1
    need "${LBA_PAIRED}/finetuned" "stage e3_ft_validate" || return 1
    # Did finetuning on LBA move the posteriors for the three families the model was
    # originally trained on? bench5 ONLY: those are the cases both arms load from the
    # shared benchmark files. The script's default list adds mixed_predictor, which each
    # arm simulates itself -- same seeded parameters, DIFFERENT datasets -- and --strict
    # compares only true_set, so it would pass while comparing unpaired data.
    run "forgetting study — paired C2ST original vs finetuned" -- \
        "$PY" -m experiments.experiment_3.model_class_paired_c2st \
            --orig_dir "${LBA_PAIRED}/original/predictions" \
            --ft_dir "${LBA_PAIRED}/finetuned/predictions" \
            --outdir "${LBA_PAIRED}/c2st" \
            --scenarios "${GAP_CASES[@]}" \
            "${C2ST_ARGS[@]}" \
            --max_num_categories "$BENCH_MAX_CAT" --strict
}

# LBA on the empirical lexical-decision data, STIMULUS-CODED (favored_idx per trial)
# with the condition_only design (intercept + condition). That design is the one the
# PPC itself argued for: dropping u_2/u_3/u_1:u_2 cut the nonword median-RT error from
# 266 ms to 18 ms while keeping the observed Speed-Accuracy gap. Lexicality needs no
# regressor under stimulus coding -- it reaches the model as favored_idx.
LBA_RD_DESIGN="condition_only"
LBA_RD_POST="./outputs/data/real_data/lba_stim_cond/mc/lba_mc_posteriors.npz"
LBA_RD_PPC="./outputs/figures/real_data/ppc/lba_stim_cond"

marker_e3_ppc_apply() { echo "$LBA_RD_POST"; }
stage_e3_ppc_apply() {
    need "${LBA_FT_CKPT}.done" "stage e3_finetune" || return 1
    need "./data/data_application_lba.csv" "the empirical LBA dataset (not produced here)" || return 1
    # The producing scripts stay in experiment_2/ (ppc.py and data_utils.py serve all
    # four families); only the RESULTS belong to experiment 3. The FINETUNED
    # checkpoint, since only it has an LBA decoder worth applying.
    run "LBA real-data apply (stimulus-coded, ${LBA_RD_DESIGN})" -- \
        "$PY" -m experiments.experiment_2.lba_mc_cf_apply \
            "${MC_SAMPLER[@]}" \
            --checkpoint "$LBA_FT_CKPT" \
            --lba_fixed_A "$LBA_FIXED_A" --design "$LBA_RD_DESIGN" \
            --outdir "$(dirname "$LBA_RD_POST")/" \
            --encoder_num_layers "$ENC_LAYERS" --decoder_num_layers "$DEC_LAYERS" \
            --encoder_num_heads "$ENC_HEADS" --decoder_num_heads "$HEADS" \
            --encoder_embed_dim "$ENC_EMBED" \
            --proj_dim "$PROJ" --num_seeds "$SEEDS" --seed_dim "$MC_DIM" \
            --model_embed_dim 8 --num_models 4 \
            --max_num_regressors "$MC_MAX_REG" --max_num_categories "$MC_MAX_CAT"
}

# Real-data-design validation for LBA: same script and rules as e2_validate, against
# the FINETUNED checkpoint and the design the real-data fit uses (LBA_RD_DESIGN), with
# each dataset borrowing a real subject's condition coding, trial count and per-trial
# word/nonword stimulus (favored_idx).
marker_e3_realdata_validate() { echo "${REALDATA_PRED}/lba_realdata_cf_pred.npz"; }
stage_e3_realdata_validate() {
    need "${LBA_FT_CKPT}.done" "stage e3_finetune" || return 1
    need "./data/data_application_lba.csv" "the empirical LBA dataset (not produced here)" || return 1
    run "real-data-design validation — lba (${LBA_RD_DESIGN})" -- \
        "$PY" -m experiments.experiment_2.realdata_validate \
            "${MC_ENC[@]}" "${MC_SAMPLER[@]}" \
            --model lba --design "$LBA_RD_DESIGN" --lba_fixed_A "$LBA_FIXED_A" \
            --checkpoint "$LBA_FT_CKPT" --num_models 4 \
            --outdir "$REALDATA_FIG" --pred_dir "$REALDATA_PRED" \
            --num_datasets 200 --num_samples 200 \
            --seed "$VAL_SEED" \
            --max_num_regressors "$MC_MAX_REG" --max_num_categories "$MC_MAX_CAT"
}

marker_e3_ppc() { echo "${LBA_RD_PPC}_conventional"; }
stage_e3_ppc() {
    need "$LBA_RD_POST" "stage e3_ppc_apply" || return 1
    # TWO variants from the SAME posterior draws, differing only in what the forward
    # simulation holds fixed. Which belongs in the paper is still open, so both are
    # produced and finalize_experiment_3 files them under ppc/free/ and
    # ppc/conventional/. --design MUST match the fit; ppc.py checks it against the
    # design_name stored in the posteriors file.
    run "LBA PPC — free (s_v, s_tau as estimated)" -- \
        "$PY" -m experiments.experiment_2.ppc \
            --model lba --variant mc --design "$LBA_RD_DESIGN" \
            --posteriors "$LBA_RD_POST" --outdir "$LBA_RD_PPC" \
            --max_num_regressors "$MC_MAX_REG" \
            --max_num_categories "$MC_MAX_CAT" || return 1
    # s_v=1, s_tau=0 is the standard LBA identifiability constraint.
    run "LBA PPC — conventional (s_v=1.0, s_tau=0.0)" -- \
        "$PY" -m experiments.experiment_2.ppc \
            --model lba --variant mc --design "$LBA_RD_DESIGN" \
            --posteriors "$LBA_RD_POST" --outdir "${LBA_RD_PPC}_conventional" \
            --fix_natural "s_v=1.0,s_tau=0.0" \
            --max_num_regressors "$MC_MAX_REG" \
            --max_num_categories "$MC_MAX_CAT"
}

marker_e3_finalize() { echo ""; }
stage_e3_finalize() {
    run "assemble outputs/finals/experiment-3" -- \
        "$PY" -m experiments.finalize_experiment_3
}

# Re-assembles experiment-3 once the LBA amortization gap exists. e3_finalize runs in the
# main sequence BEFORE the gap stages and simply leaves gap/ out (finalize copies only the
# sources that exist, and never deletes), so this second pass adds gap/ in place.
marker_e3_gap_finalize() { echo ""; }
stage_e3_gap_finalize() {
    run "re-assemble outputs/finals/experiment-3 with gap/" -- \
        "$PY" -m experiments.finalize_experiment_3
}

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4 — encoder representation (aDDM)
# ══════════════════════════════════════════════════════════════════════════════
# FINAL SCOPE (2026-09-27): the 5-parameter vsvfixed aDDM only (v and s_v pinned;
# free: v_diff, omega, a, tau, s_tau), plus the 5-family pooled geometry. The other
# variants (intercept_only_7d, vfixed, svfixed, varying_values) are out of the paper.
#
# Called DIRECTLY against experiments/experiment_4/. The older run_addm_*.sh launchers
# imported experiments/encoder_rep/, whose copies stopped being updated on 2026-09-08..12
# while experiment_4's moved on; both were removed on 2026-09-28 (still in git history).
#
# Every aDDM call below MUST carry the same E4_VARIANT flags: they select the
# simulated model (resolve_case), so a mismatch trains and validates on different
# generative models -- the empty-fixed_values failure mode again.
E4_VARIANT=(--case intercept_only --first_fixation item0 --fix_v --fix_s_v)
E4_TAG="vsvfixed_e500"                 # baseline arm's run tag
E4_EPOCHS=500                          # == the budget both arms were matched at
E4_VAL=(--val_batch_size 500 --num_samples 500 --sample_chunk_size 50)
E4_DATA="./outputs/data/encoder_rep"
E4_FIG="./outputs/figures/encoder_rep"
E4_BASELINE_CKPT="./outputs/checkpoints/encoder_rep/bf_baseline/addm_vsvfixed/intercept_only/item0_${E4_TAG}/addm_baseline_fm/model.keras"

marker_e4_probe() { echo "${E4_DATA}/addm_vsvfixed/intercept_only/addm_encoder_rep_data.npz"; }
stage_e4_probe() {
    need "${MC_CKPT}.done" "stage e1_train" || return 1
    # FROZEN arm: the model-class encoder, frozen, feeding a FlowMatching head. Always
    # retrained -- its checkpoint path does not encode which encoder it was built on,
    # so a leftover from an older model-class checkpoint would pass silently.
    local mode
    for mode in train validate; do
        run "aDDM frozen-encoder arm — ${mode}" -- \
            "$PY" -m experiments.experiment_4.addm_bf_apply \
                "${MC_ENC[@]}" \
                --mode "$mode" "${E4_VARIANT[@]}" \
                --num_models 3 --encoder_device cuda --epochs "$E4_EPOCHS" \
                "${E4_VAL[@]}" --cogformer_checkpoint "$MC_CKPT" || return 1
    done
    run "aDDM frozen-encoder diagnostics" -- \
        "$PY" -m experiments.experiment_4.addm_diagnostics \
            --case intercept_only --arm frozen \
            --data "${E4_DATA}/addm_vsvfixed/intercept_only/addm_encoder_rep_data.npz" || return 1
    # BASELINE arm: learned SetTransformer summary. It does not depend on CogFormer,
    # so the kept checkpoint (trained 2026-09 under BayesFlow 2.0.13; loads with
    # identical weights under 2.0.14) is reused unless FORCE is set. It trains through
    # the synchronous fit_online path, so the keras fork-RNG bug never touched it.
    if [[ ! -e "$E4_BASELINE_CKPT" || -n "${FORCE:-}" ]]; then
        run "aDDM baseline arm — train" -- \
            "$PY" -m experiments.experiment_4.addm_bf_baseline \
                --mode train "${E4_VARIANT[@]}" --tag "$E4_TAG" --subnet_dropout 0 \
                --epochs "$E4_EPOCHS" "${E4_VAL[@]}" || return 1
    else
        echo "    · baseline checkpoint present, reused: $E4_BASELINE_CKPT"
    fi
    run "aDDM baseline arm — validate" -- \
        "$PY" -m experiments.experiment_4.addm_bf_baseline \
            --mode validate "${E4_VARIANT[@]}" --tag "$E4_TAG" --subnet_dropout 0 \
            --epochs "$E4_EPOCHS" "${E4_VAL[@]}" || return 1
    # Redraw the baseline's four diagnostics with the SAME renderer as the frozen arm,
    # so the two arms' figures differ only in their arm colour. Overwrites the
    # baseline script's own default-styled plots under the same filenames.
    run "aDDM baseline-arm diagnostics" -- \
        "$PY" -m experiments.experiment_4.addm_diagnostics \
            --case intercept_only --arm baseline \
            --data "${E4_FIG}/addm_baseline_vsvfixed/intercept_only/item0_${E4_TAG}/addm_baseline_data.npz"
}

marker_e4_compare() { echo "${E4_DATA}/addm_compare/intercept_only_vsvfixed/shared_validation.npz"; }
stage_e4_compare() {
    need "${E4_DATA}/addm_vsvfixed/intercept_only/addm_encoder_rep_data.npz" "stage e4_probe" || return 1
    # PAIRED: both arms re-scored on one shared validation set, then C2ST against an
    # empirical draw-split null, log-density and CRPS. Separate validate runs simulate
    # separate datasets, which is why this stage exists at all.
    run "aDDM arm comparison (shared validation C2ST)" -- \
        "$PY" -m experiments.experiment_4.addm_arm_compare \
            "${MC_ENC[@]}" \
            "${E4_VARIANT[@]}" --tag "$E4_TAG" \
            --cogformer_checkpoint "$MC_CKPT" --num_models 3 || return 1
    run "aDDM combined recovery (frozen vs baseline)" -- \
        "$PY" -m experiments.experiment_4.combined_recovery \
            --frozen_data "${E4_DATA}/addm_vsvfixed/intercept_only/addm_encoder_rep_data.npz" \
            --baseline_data "${E4_FIG}/addm_baseline_vsvfixed/intercept_only/item0_${E4_TAG}/addm_baseline_data.npz"
}

marker_e4_representation() { echo "${E4_DATA}/representation/crossfamily/app3_summaries.npz"; }
stage_e4_representation() {
    need "${MC_CKPT}.done" "stage e1_train" || return 1
    # REPRESENTATION level, not posterior level: RSA (crossnobis) + CKA against a
    # dimensionality-matched null. The frozen summary is 128-d and the learned one
    # 16-d, so no posterior metric can compare them.
    # The pooled geometry draws one panel per encoder, before and after the LBA
    # finetune, so this stage also needs e3_finetune. It saves its raw simulated
    # datasets (app3_raw_data.npz): --seed alone cannot regenerate them.
    need "${LBA_FT_CKPT}.done" "stage e3_finetune" || return 1
    run "cross-family pooled geometry (5 families, before/after LBA finetune)" -- \
        "$PY" -m experiments.experiment_4.crossfamily_representation \
            "${MC_ENC[@]}" \
            --cogformer_checkpoint "$MC_CKPT" --num_models 3 \
            --finetuned_checkpoint "$LBA_FT_CKPT" --finetuned_num_models 4 \
            --fixed_A "$LBA_FIXED_A" || return 1
    run "aDDM representation — frozen vs baseline (vsvfixed)" -- \
        "$PY" -m experiments.experiment_4.addm_representation \
            "${MC_ENC[@]}" \
            "${E4_VARIANT[@]}" --tag "$E4_TAG" \
            --cogformer_checkpoint "$MC_CKPT" --num_models 3 || return 1
    # The combined page reads the two summaries it plots from their SOURCES here; its
    # defaults point into outputs/finals/, which would plot the previous run.
    run "combined panel (probe + C2ST + representation)" -- \
        "$PY" -m experiments.experiment_4.combined_panel \
            --geometry_npz "${E4_DATA}/representation/crossfamily/app3_summaries.npz" \
            --rep_npz "${E4_DATA}/representation/intercept_only_vsvfixed/summaries.npz"
}

marker_e4_finalize() { echo ""; }
stage_e4_finalize() {
    run "assemble outputs/finals/experiment-4" -- \
        "$PY" -m experiments.finalize_experiment_4
}

# ══════════════════════════════════════════════════════════════════════════════
# stage table
# ══════════════════════════════════════════════════════════════════════════════
# The amortization gap (BayesFlow baselines + gap C2STs) is fully separable: no other stage
# reads its outputs. It runs LAST, so the main sequence (exp1 -> exp4) never waits on 20
# baseline trainings. gap_train needs only e1_data and can run CONCURRENTLY on another GPU
# or host; the gap C2STs then need e1_validate (e1a) and e3_ft_validate (e3).
ALL_STAGES=(
    e1_data e1_train e1_validate e1_ensemble e1_benchmark e1_design_eig e1_finalize
    e2_apply e2_refit_svfix e2_ppc e2_validate e2_finalize
    e3_zeroshot e3_finetune e3_ft_validate e3_original e3_paired_c2st
    e3_compare e3_ppc_apply e3_ppc e3_realdata_validate e3_finalize
    e4_probe e4_compare e4_representation e4_finalize
    e1a_bf e3_bf e1a_c2st e3_gap_c2st e1a_finalize e3_gap_finalize
)
GROUP_exp1="e1_data e1_train e1_validate e1_ensemble e1_benchmark e1_design_eig e1_finalize"
GROUP_exp2="e2_apply e2_refit_svfix e2_ppc e2_validate e2_finalize"
GROUP_exp3="e3_zeroshot e3_finetune e3_ft_validate e3_original e3_paired_c2st e3_compare e3_ppc_apply e3_ppc e3_realdata_validate e3_finalize"
GROUP_exp4="e4_probe e4_compare e4_representation e4_finalize"
GROUP_main="$GROUP_exp1 $GROUP_exp2 $GROUP_exp3 $GROUP_exp4"
GROUP_gap_train="e1a_bf e3_bf"
GROUP_gap_eval="e1a_c2st e3_gap_c2st e1a_finalize e3_gap_finalize"
GROUP_gap="$GROUP_gap_train $GROUP_gap_eval"
GROUP_exp1a="e1a_bf e1a_c2st e1a_finalize"
GROUP_finalize="e1_finalize e1a_finalize e2_finalize e3_finalize e4_finalize"

expand_group() {
    case "$1" in
        exp1) echo "$GROUP_exp1" ;;  exp1a) echo "$GROUP_exp1a" ;;
        exp2) echo "$GROUP_exp2" ;;
        exp3) echo "$GROUP_exp3" ;;  exp4) echo "$GROUP_exp4" ;;
        main) echo "$GROUP_main" ;;
        gap) echo "$GROUP_gap" ;;  gap_train) echo "$GROUP_gap_train" ;;  gap_eval) echo "$GROUP_gap_eval" ;;
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
    echo "groups: main (exp1 exp2 exp3 exp4)  gap (gap_train gap_eval)  exp1a  finalize  all"
}

# ── selection ─────────────────────────────────────────────────────────────────
if [[ "${1:-}" == remaining_two_gpu && $# -eq 1 ]]; then
    exec "$PY" -m experiments.continue_study --launcher "$PROJECT_ROOT/run_experiments.sh" \
        --gpu0 "$GPU0" --gpu1 "$GPU1"
fi
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
echo " study  : ${STUDY_ROOT}   (generator ${DESIGN_GENERATOR}, MC_TAG ${MC_TAG})"
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
