#!/usr/bin/env bash
# run_amortization_gap.sh — the BayesFlow amortization-gap evaluation for DDM and CDM.
#
# The gap is a paired C2ST: for each bench5 case, how distinguishable are the
# posteriors of the case-specific BayesFlow baseline from those of the single
# amortized ModelClass CogFormer? 0.5 = indistinguishable (no gap).
#
# THREE STAGES, strictly ordered — stage B consumes stage A's output.
#
#   A  bf_validate    BF posterior draws          -> outputs/data/model_family/{m}_{case}_data.npz
#   B  cf_validate    CogFormer draws, PAIRED     -> $PRED_DIR/{m}_{case}_cf_pred.npz
#   C  c2st           the comparison              -> outputs/figures/c2st/{m}/...
#
# WHY PAIRED MATTERS. C2ST scores two sets of posterior draws over the SAME
# simulated datasets. Stage B therefore re-runs CogFormer validation with
# --data_dir pointing at stage A's npz files, which makes it load BF's simulated
# rts/choices/design_matrices instead of simulating its own. The pre-existing
# predictions in outputs/data/model_class/ (2026-06-21) were NOT produced this
# way, and model_family_c2st's only guard is a shape check on the parameter grid
# (both arms are 181x7 regardless), so feeding it those files yields a
# silently-wrong, inflated gap. They are left untouched; stage B writes to its
# own PRED_DIR.
#
# GPU SPLIT. model_class_cf_validate has no --models flag, but in paired mode it
# skips any case whose npz is absent. So stage B is split across both cards by
# giving each process a view directory containing only one model's npz files.
# Both write into the same PRED_DIR -- filenames are disjoint (ddm_* vs cdm_*)
# and seeding is per-(model,case), so the split does not change any result.
#
# Usage: setsid nohup bash outputs/run_amortization_gap.sh > outputs/logs/amortization_gap.log 2>&1 &
set -uo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PY=.venv/bin/python
source experiments/shared_config.sh

# MPLBACKEND=Agg is REQUIRED, not a tidiness flag. This desktop session is Wayland
# with Xwayland, and the inherited DISPLAY=:0 no longer accepts connections, so
# matplotlib's backend auto-resolution blocks forever on the X unix socket -- the
# process parks in unix_wait_for_peer at 0% CPU, with no error and no timeout.
# Every stage here plots, so without this the pipeline hangs on its FIRST case,
# after writing the npz and metrics CSV but before any figure. Verified 2026-09-08:
# a bare `matplotlib.get_backend()` in this venv never returns; under Agg, instant.
export MPLBACKEND=Agg

MODELS=(ddm cdm)
CASES=(intercept_only fixed regressed fixed_regressed interaction)

# Geometry: the CogFormer arm is the r4c4 ModelClass checkpoint, so the C2ST grid
# is r4c4 (181 rows). BF trained at r2c2 max, which pads into r4c4 cleanly.
PRED_DIR="./outputs/data/model_family_paired"
FIG_DIR="./outputs/figures/model_class/cf_paired/"
BF_DATA_DIR="./outputs/data/model_family"
SPLIT_ROOT="./outputs/data/_mf_split"

log() { echo "[$(date '+%F %T')] $*"; }

# ── stage A: BF validation, one model per GPU ─────────────────────────────────
bf_validate_model() {
    local model=$1 gpu=$2 cpus=$3
    for case in "${CASES[@]}"; do
        local out="outputs/logs/bfval_${model}_${case}.log"
        log "[A/$model] validate case=$case -> $out"
        # PREALLOCATE=false: jax otherwise grabs 75% of the 49GB card. Shared box.
        taskset -c "$cpus" env CUDA_VISIBLE_DEVICES="$gpu" \
            XLA_PYTHON_CLIENT_PREALLOCATE=false \
            "$PY" -m experiments.model_family.model_family_bf_validate \
            --model "$model" --case "$case" \
            --batch_size "$VAL_BATCH" --num_samples 200 \
            --skip_posteriors > "$out" 2>&1
        local rc=$?
        [ $rc -eq 0 ] && log "[A/$model] done case=$case" \
                      || log "[A/$model] FAILED case=$case (exit $rc) -- see $out; continuing"
    done
}

log "=== stage A: BF validation (ddm->GPU0, cdm->GPU1) ==="
bf_validate_model ddm 0 0-23 &
pid_ddm=$!
bf_validate_model cdm 1 24-47 &
pid_cdm=$!
wait $pid_ddm $pid_cdm
log "stage A complete"

# Stage B is pointless for a model whose npz files are missing, and a partial
# stage A must not silently produce a partial gap matrix. Report and continue
# with whatever is present, but say so loudly.
missing=0
for m in "${MODELS[@]}"; do
    for c in "${CASES[@]}"; do
        [ -f "$BF_DATA_DIR/${m}_${c}_data.npz" ] || { log "MISSING $BF_DATA_DIR/${m}_${c}_data.npz"; missing=$((missing+1)); }
    done
done
[ $missing -gt 0 ] && log "WARNING: $missing/10 BF npz files missing; the gap matrix will be incomplete"

# ── stage B: paired CogFormer validation, one model per GPU ───────────────────
mkdir -p "$PRED_DIR" "$FIG_DIR"
rm -rf "$SPLIT_ROOT"; mkdir -p "$SPLIT_ROOT/ddm" "$SPLIT_ROOT/cdm"
for m in "${MODELS[@]}"; do
    for f in "$BF_DATA_DIR/${m}_"*_data.npz; do
        [ -e "$f" ] && ln -sf "$(realpath "$f")" "$SPLIT_ROOT/$m/$(basename "$f")"
    done
done

cf_validate_model() {
    local model=$1 gpu=$2 cpus=$3
    local out="outputs/logs/cfval_paired_${model}.log"
    log "[B/$model] paired validate on GPU$gpu -> $out"
    # samples_chunk_size 1: peak memory scales with batch_size * chunk, and this
    # box is shared -- chunk 2 has OOMed here before.
    taskset -c "$cpus" env CUDA_VISIBLE_DEVICES="$gpu" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$PY" -m experiments.model_class.model_class_cf_validate \
        --self_attn_padding_mask \
        --encoder_num_layers "$ENC_LAYERS" --decoder_num_layers "$DEC_LAYERS" \
        --encoder_num_heads "$HEADS" --decoder_num_heads "$HEADS" \
        --proj_dim "$PROJ" --num_seeds "$SEEDS" --seed_dim "$MC_DIM" \
        --model_embed_dim 8 \
        --max_num_regressors "$MC_MAX_REG" --max_num_categories "$MC_MAX_CAT" \
        --checkpoint "$MC_CKPT" \
        --data_dir "$SPLIT_ROOT/$model" \
        --pred_dir "$PRED_DIR" --outdir "$FIG_DIR" \
        --batch_size "$VAL_BATCH" --num_samples 200 --num_obs "$OBS" \
        --samples_chunk_size 1 --skip_posteriors > "$out" 2>&1
    local rc=$?
    [ $rc -eq 0 ] && log "[B/$model] done" || log "[B/$model] FAILED (exit $rc) -- see $out"
}

log "=== stage B: paired CogFormer validation ==="
cf_validate_model ddm 0 0-23 &
pid_ddm=$!
cf_validate_model cdm 1 24-47 &
pid_cdm=$!
wait $pid_ddm $pid_cdm
log "stage B complete"

# ── stage C: the C2ST itself ──────────────────────────────────────────────────
# CPU-bound and quick. Only BF_vs_Original is reachable: Original_vs_Finetuned
# additionally needs --ft_pred_dir from the LBA-finetuned checkpoint.
log "=== stage C: C2ST (BF_vs_Original) ==="
for m in "${MODELS[@]}"; do
    out="outputs/logs/c2st_${m}.log"
    log "[C/$m] c2st -> $out"
    "$PY" -m experiments.model_family.model_family_c2st \
        --model "$m" \
        --bf_data_dir "$BF_DATA_DIR" \
        --mc_pred_dir "$PRED_DIR" \
        --scenarios BF_vs_Original \
        --max_num_categories "$MC_MAX_CAT" > "$out" 2>&1
    rc=$?
    [ $rc -eq 0 ] && log "[C/$m] done" || log "[C/$m] FAILED (exit $rc) -- see $out"
done

log "=== all stages complete ==="
for m in "${MODELS[@]}"; do
    s="./outputs/figures/c2st/${m}/${m}_c2st_summary.csv"
    [ -f "$s" ] && { log "summary $m:"; cat "$s"; } || log "no summary for $m"
done
