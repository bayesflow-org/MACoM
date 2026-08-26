#!/usr/bin/env bash
# run_pipeline.sh — end-to-end orchestration of the CogFormer-2 experiment suite,
# in dependency order. One entry point, one config source, one log tree.
#
# This supersedes ad-hoc invocation of the per-directory run.sh scripts for a full
# reproduction. Those still work standalone; this wires them in the right order,
# pins the design geometry they do not pin themselves (see GEOMETRY below), and
# refuses to start a stage whose inputs are absent.
#
# STAGES
#   1  mc_train      ModelClass CogFormer (DDM+RDM+CDM), r4c4, batch 256, 10000 epochs
#   2  mc_validate   6 validation scenarios x 3 model families
#   3  mc_ensemble   random-design ensemble diagnostics for the same checkpoint
#   4  real_data     DDM/RDM/CDM apply (mf + mc) -> PPC -> posterior summary figures
#   5  lba_zeroshot  LBA zero-shot inference (family never seen in training)
#   6  lba_finetune  LBA finetune -> validate -> zero-shot/finetuned compare ->
#                    paired C2ST forgetting study (amortization gap on DDM/RDM/CDM)
#   7  encoder_rep   aDDM encoder-representation probe: frozen-encoder arm,
#                    BayesFlow SetTransformer baseline, diagnostics
#
# USAGE
#   bash experiments/run_pipeline.sh                    # every stage, in order
#   bash experiments/run_pipeline.sh 4 5 6              # only these stages
#   bash experiments/run_pipeline.sh mc_train real_data # by name
#   DRY_RUN=1 bash experiments/run_pipeline.sh          # print commands, run nothing
#   SKIP_PREFLIGHT=1 ...                                # bypass input checks
#
# ENVIRONMENT (forwarded where the sub-scripts understand them)
#   DRY_EPOCHS, USE_WANDB, NUM_SIM_WORKERS, SELF_ATTN_PADDING_MASK
#
# GEOMETRY — the reason several stages are inlined here rather than delegated:
# experiments/utils.py defaults --max_num_regressors/--max_num_categories to 5,
# i.e. r5c5, and model_class/run.sh never overrides them. But MC_CKPT, every apply
# script and the r4c4_b256_e10000 checkpoint directory all assume r4c4. The
# checkpoint filename does NOT encode r/c, so an r5c5 run silently produces a file
# with the right name and an incompatible encoder (build_encoder_input_dim: r4c4
# -> 72, r5c5 -> 186), which then fails to load or, worse, loads at the wrong
# width. Every stage below passes MC_MAX_REG/MC_MAX_CAT explicitly.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
source "$SCRIPT_DIR/shared_config.sh"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1

# ── ModelClass training scale ──────────────────────────────────────────────────
# Literal 256, not MC_BATCH (64): MC_CKPT hardcodes _b256_ because the production
# run used 256. MC_TAG must match the directory component of MC_CKPT.
MC_TRAIN_BATCH=256
MC_TAG="r4c4_b256_e10000"
MC_CKPT_DIR="./outputs/checkpoints/model_class/cf/${MC_TAG}/"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/pipeline/${RUN_ID}"
mkdir -p "$LOG_DIR"

MASK_ARGS=()
if [[ "${SELF_ATTN_PADDING_MASK:-true}" == "true" ]]; then
    MASK_ARGS=(--self_attn_padding_mask)
else
    MASK_ARGS=(--no-self_attn_padding_mask)
fi
DRY=(); [[ -n "${DRY_EPOCHS:-}" ]] && DRY=(--dry_run_epochs "$DRY_EPOCHS")
WANDB=(); [[ -n "${USE_WANDB:-}" ]] && WANDB=(--use_wandb)
SIMW=(); [[ -n "${NUM_SIM_WORKERS:-}" ]] && SIMW=(--num_sim_workers "$NUM_SIM_WORKERS")

# Architecture block shared by train / validate / ensemble. These MUST agree:
# the padding-mask flag changes the forward pass without changing state_dict keys,
# so a mismatch is silent.
MC_ARCH=(
    "${MASK_ARGS[@]}"
    --encoder_num_layers "$ENC_LAYERS" --decoder_num_layers "$DEC_LAYERS"
    --encoder_num_heads  "$HEADS"      --decoder_num_heads  "$HEADS"
    --proj_dim "$PROJ" --num_seeds "$SEEDS" --seed_dim "$MC_DIM"
    --model_embed_dim 8
)
MC_GEOM=(--max_num_regressors "$MC_MAX_REG" --max_num_categories "$MC_MAX_CAT")

# ── plumbing ───────────────────────────────────────────────────────────────────
run() {                      # run <stage> <description> -- <command...>
    local stage="$1" desc="$2"; shift 3
    local log="${LOG_DIR}/${stage}.log"
    echo "--- [${stage}] ${desc}"
    if [[ -n "${DRY_RUN:-}" ]]; then printf '      %q ' "$@"; echo; return 0; fi
    local t0=$SECONDS
    "$@" >>"$log" 2>&1
    local rc=$?
    local mins=$(( (SECONDS - t0) / 60 ))
    if [[ $rc -ne 0 ]]; then
        echo "!!! [${stage}] FAILED (exit $rc) after ${mins}m — see $log"
        tail -20 "$log" | sed 's/^/    | /'
        return $rc
    fi
    echo "    [${stage}] ok (${mins}m)"
}

need_file() {                # need_file <path> <what produces it>
    [[ -f "$1" ]] && return 0
    echo "!!! missing input: $1"
    echo "    produced by: $2"
    return 1
}

# ── stages ─────────────────────────────────────────────────────────────────────
stage_mc_train() {
    run mc_train "ModelClass CF train (r${MC_MAX_REG}c${MC_MAX_CAT}, b${MC_TRAIN_BATCH}, e${EPOCHS})" -- \
        python -m experiments.model_class.model_class_cf_train \
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

stage_mc_validate() {
    need_file "$MC_CKPT" "stage 1 (mc_train)" || return 1
    run mc_validate "ModelClass CF validate — 6 scenarios x 3 families" -- \
        python -m experiments.model_class.model_class_cf_validate \
            "${MC_ARCH[@]}" "${MC_GEOM[@]}" \
            --checkpoint "$MC_CKPT" \
            --batch_size 200 --num_samples 200 --num_obs "$OBS" \
            --samples_chunk_size 1
}

stage_mc_ensemble() {
    need_file "$MC_CKPT" "stage 1 (mc_train)" || return 1
    # samples_chunk_size 1: chunk 2 OOMs the RTX 5000 Ada on ensemble sampling.
    run mc_ensemble "ModelClass random-design ensemble" -- \
        python -m experiments.model_class.model_class_cf_ensemble \
            "${MC_ARCH[@]}" "${MC_GEOM[@]}" \
            --checkpoint "$MC_CKPT" \
            --batch_size 200 --num_obs "$OBS" --num_samples 200 \
            --samples_chunk_size 1 --families "DDM,RDM,CDM"
}

stage_real_data() {
    need_file "$MC_CKPT" "stage 1 (mc_train)" || return 1
    run real_data "apply trained checkpoints to empirical data" -- \
        bash "$SCRIPT_DIR/real_data/run.sh" || return 1
    for model in ddm rdm cdm; do
        for variant in mf mc; do
            run real_data "PPC ${model} ${variant}" -- \
                python -m experiments.real_data.ppc \
                    --model "$model" --variant "$variant" \
                    --max_num_regressors "$MC_MAX_REG" \
                    --max_num_categories "$MC_MAX_CAT" || return 1
        done
    done
    run real_data "posterior summary figures (mc vs mf)" -- \
        python -m experiments.real_data.plot_posteriors
}

stage_lba_zeroshot() {
    need_file "$MC_CKPT" "stage 1 (mc_train)" || return 1
    # SKIP_COMPARE too: the compare step consumes finetuned predictions that do
    # not exist until stage 6, and run_lba.sh runs it unguarded otherwise.
    SKIP_FINETUNE=1 SKIP_FT_VALIDATE=1 SKIP_COMPARE=1 \
        run lba_zeroshot "LBA zero-shot (A=${LBA_FIXED_A})" -- \
        bash "$SCRIPT_DIR/model_class/run_lba.sh"
}

stage_lba_finetune() {
    need_file "$MC_CKPT" "stage 1 (mc_train)" || return 1
    SKIP_ZERO_SHOT=1 \
        run lba_finetune "LBA finetune + validate + compare (A=${LBA_FIXED_A})" -- \
        bash "$SCRIPT_DIR/model_class/run_lba.sh" || return 1
    # The forgetting arm: did finetuning on LBA move the posteriors for the three
    # families the model was originally trained on? Paired, so no re-simulation.
    run lba_finetune "forgetting study — paired C2ST original vs finetuned" -- \
        python -m experiments.model_class.model_class_paired_c2st \
            --max_num_categories "$MC_MAX_CAT"
}

stage_encoder_rep() {
    need_file "$MC_CKPT" "stage 1 (mc_train)" || return 1
    # Frozen-encoder arm: the encoder is the 3-family ModelClass one, which never
    # saw aDDM. num_models 3 must match that checkpoint, not the 4-model encoder_rep one.
    for mode in train validate; do
        run encoder_rep "aDDM frozen-encoder arm — ${mode}" -- \
            python -m experiments.encoder_rep.addm_bf_apply \
                --mode "$mode" --case intercept_only --first_fixation item0 \
                --num_models 3 --cogformer_checkpoint "$MC_CKPT" \
                "${MASK_ARGS[@]}" || return 1
    done
    run encoder_rep "aDDM frozen-encoder diagnostics" -- \
        python -m experiments.encoder_rep.addm_diagnostics || return 1
    # Learned-summary baseline, same simulator/design/budget, its own tag so the
    # two arms' outputs cannot collide.
    for mode in train validate; do
        run encoder_rep "aDDM BayesFlow baseline — ${mode}" -- \
            python -m experiments.encoder_rep.addm_bf_baseline \
                --mode "$mode" --case intercept_only --first_fixation item0 \
                --tag best_defaults || return 1
    done
}

ALL_STAGES=(mc_train mc_validate mc_ensemble real_data lba_zeroshot lba_finetune encoder_rep)

# ── stage selection ────────────────────────────────────────────────────────────
SELECTED=()
if [[ $# -eq 0 ]]; then
    SELECTED=("${ALL_STAGES[@]}")
else
    for arg in "$@"; do
        if [[ "$arg" =~ ^[1-7]$ ]]; then
            SELECTED+=("${ALL_STAGES[$((arg-1))]}")
        else
            found=""
            for s in "${ALL_STAGES[@]}"; do [[ "$s" == "$arg" ]] && found=1 && SELECTED+=("$s"); done
            [[ -z "$found" ]] && { echo "unknown stage: $arg"; echo "valid: ${ALL_STAGES[*]}"; exit 2; }
        fi
    done
fi

# ── preflight ──────────────────────────────────────────────────────────────────
preflight() {
    local problems=0
    echo "--- preflight"
    for s in "${SELECTED[@]}"; do
        case "$s" in
            real_data)
                # real_data/run.sh applies BOTH the model-family and model-class
                # checkpoints for all three models. The mf arm needs three separate
                # per-family checkpoints, which are trained OUTSIDE this pipeline
                # (experiments/model_family/run.sh) -- see the note in the summary.
                for v in DDM_MF_CKPT RDM_MF_CKPT CDM_MF_CKPT; do
                    [[ -f "${!v}" ]] || { echo "    ! $v missing: ${!v}"; problems=1; }
                done
                [[ -d ./data ]] || { echo "    ! ./data missing (empirical datasets)"; problems=1; }
                ;;
            lba_finetune)
                [[ -f "$LBA_FT_CKPT" ]] || echo "    i LBA finetune checkpoint absent; this stage will create it"
                ;;
        esac
    done
    if [[ $problems -ne 0 ]]; then
        echo "    preflight found missing inputs (set SKIP_PREFLIGHT=1 to run anyway)"
        return 1
    fi
    echo "    ok"
}

echo "============================================================"
echo " CogFormer-2 pipeline   run ${RUN_ID}"
echo " stages: ${SELECTED[*]}"
echo " logs:   ${LOG_DIR}/"
[[ -n "${DRY_RUN:-}" ]] && echo " DRY RUN — no commands will execute"
echo "============================================================"

if [[ -z "${SKIP_PREFLIGHT:-}" && -z "${DRY_RUN:-}" ]]; then
    preflight || exit 1
fi

PIPE_T0=$SECONDS
for s in "${SELECTED[@]}"; do
    echo
    echo "=== stage: $s ==="
    if ! "stage_${s}"; then
        echo
        echo "PIPELINE ABORTED at stage '$s' after $(( (SECONDS - PIPE_T0) / 60 ))m"
        echo "Logs: ${LOG_DIR}/"
        exit 1
    fi
done

echo
echo "============================================================"
echo " PIPELINE COMPLETE in $(( (SECONDS - PIPE_T0) / 60 ))m — logs in ${LOG_DIR}/"
echo "============================================================"
