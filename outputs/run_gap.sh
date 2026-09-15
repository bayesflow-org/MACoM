#!/usr/bin/env bash
# Amortization gap (BF_vs_Original C2ST) for the __e2000new BayesFlow baselines.
#
# The gap is a C2ST between the CogFormer model-class posteriors and the BayesFlow
# posteriors ON THE SAME DATASETS, so it needs three stages:
#   1. stage   the e2000new npz under the untagged name the downstream scripts expect
#   2. cf_validate  --data_dir <staged npz>  -> CogFormer predictions on those datasets
#   3. c2st         BF_vs_Original           -> per-case accuracy + summary
#
# WHY STAGING. model_class_cf_validate and model_family_c2st both build their input
# path as "<dir>/<model>_<case>_data.npz" -- neither has a --tag. Pointed at the real
# data dir they would silently pick up the UNTAGGED production npz (2000 ep, OLD config)
# and report a gap for the wrong checkpoints. Staging symlinks named the way they expect,
# in a directory that contains nothing else, makes that impossible.
#
# NOTHING PRODUCTION IS TOUCHED: predictions, figures and C2ST output all go to
# *_e2000new paths. The production c2st tree under outputs/figures/c2st/ is untouched,
# which matters because finalize_experiment_1 reads it.
#
# Usage: TAG=e200 bash outputs/run_gap.sh ddm/intercept_only cdm/intercept_only ...
#        (TAG defaults to e2000new; with no model/case args, the two intercept_only cases)
set -uo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
PY=.venv/bin/python
source experiments/shared_config.sh

TAG="${TAG:-e2000new}"
JOBS=("$@")
[ ${#JOBS[@]} -eq 0 ] && JOBS=(ddm/intercept_only cdm/intercept_only)

BF_DATA=./outputs/data/model_family
STAGE=./outputs/data/_mf_${TAG}
PRED_DIR=./outputs/data/model_family_paired_${TAG}
FIG_DIR=./outputs/figures/model_class/cf_paired_${TAG}/
C2ST_OUT=./outputs/figures/c2st_${TAG}
GPU="${GAP_GPU:-0}"
CPUS="${GAP_CPUS:-0-23}"

log() { echo "[$(date '+%F %T')] $*"; }
rm -rf "$STAGE"; mkdir -p "$STAGE" "$PRED_DIR"

# ── 1. stage ──────────────────────────────────────────────────────────────────
MODELS_SEEN=()
for job in "${JOBS[@]}"; do
    m="${job%%/*}"; c="${job##*/}"
    src="$BF_DATA/${m}_${c}__${TAG}_data.npz"
    if [ ! -f "$src" ]; then
        log "[$m/$c] SKIP -- $src not present (case not validated yet)"
        continue
    fi
    # Per-case subdir for cf_validate (it processes every case whose npz it finds, so a
    # one-case view stops it resampling cases that are not part of this round), plus a
    # flat copy for c2st, which wants one directory covering the model's cases.
    mkdir -p "$STAGE/$m/$c"
    ln -sf "$(realpath "$src")" "$STAGE/$m/$c/${m}_${c}_data.npz"
    ln -sf "$(realpath "$src")" "$STAGE/${m}_${c}_data.npz"
    log "[$m/$c] staged"
    case " ${MODELS_SEEN[*]:-} " in *" $m "*) ;; *) MODELS_SEEN+=("$m");; esac
done
[ ${#MODELS_SEEN[@]} -eq 0 ] && { log "nothing staged; exiting"; exit 1; }

RUN="taskset -c $CPUS env CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

# ── 2. paired CogFormer validation ────────────────────────────────────────────
# r4c4 geometry and MC_CKPT: a checkpoint can only be loaded at the encoder geometry it
# was trained at (build_encoder_input_dim), so these must match shared_config.
for job in "${JOBS[@]}"; do
    m="${job%%/*}"; c="${job##*/}"
    [ -d "$STAGE/$m/$c" ] || continue
    vlog="outputs/logs/gap_${TAG}_cfval_${m}_${c}.log"
    log "[$m/$c] cf_validate (paired) -> $vlog"
    $RUN "$PY" -m experiments.model_class.model_class_cf_validate \
        --self_attn_padding_mask \
        --encoder_num_layers "$ENC_LAYERS" --decoder_num_layers "$DEC_LAYERS" \
        --encoder_num_heads "$HEADS" --decoder_num_heads "$HEADS" \
        --proj_dim "$PROJ" --num_seeds "$SEEDS" --seed_dim "$MC_DIM" \
        --model_embed_dim 8 \
        --max_num_regressors "$MC_MAX_REG" --max_num_categories "$MC_MAX_CAT" \
        --checkpoint "$MC_CKPT" \
        --data_dir "$STAGE/$m/$c" --pred_dir "$PRED_DIR" --outdir "$FIG_DIR" \
        --batch_size "$VAL_BATCH" --num_samples 200 --num_obs "$OBS" \
        --samples_chunk_size 1 --skip_posteriors \
        > "$vlog" 2>&1 \
        && log "[$m/$c] cf_validate done" \
        || { log "[$m/$c] cf_validate FAILED -- see $vlog"; continue; }
done

# ── 3. C2ST: BF_vs_Original ───────────────────────────────────────────────────
for m in "${MODELS_SEEN[@]}"; do
    clog="outputs/logs/gap_${TAG}_c2st_${m}.log"
    log "[$m] c2st BF_vs_Original -> $clog"
    $RUN "$PY" -m experiments.model_family.model_family_c2st \
        --model "$m" --bf_data_dir "$STAGE" \
        --mc_pred_dir "$PRED_DIR" --scenarios BF_vs_Original \
        --max_num_categories "$MC_MAX_CAT" \
        --outdir "$C2ST_OUT/$m" \
        > "$clog" 2>&1 \
        && log "[$m] c2st done" \
        || { log "[$m] c2st FAILED -- see $clog"; continue; }
done

# ── report ────────────────────────────────────────────────────────────────────
echo
log "=== amortization gap (${TAG}) ==="
for m in "${MODELS_SEEN[@]}"; do
    s="$C2ST_OUT/$m/${m}_c2st_summary.csv"
    [ -f "$s" ] && { echo "--- $m ---"; cat "$s"; } || echo "--- $m --- no summary written"
done
