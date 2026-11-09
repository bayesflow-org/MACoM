#!/usr/bin/env bash
# setup_study.sh — prepare a directory to run the finals pipeline IN, under a design
# generator other than the main study's (e.g. gaussian_softmax).
#
#   bash experiments/setup_study.sh outputs/gaussian-softmax gaussian_softmax
#   bash experiments/setup_study.sh <dir> <generator> --reuse-main-bench-bf
#
# then run the pipeline exactly as usual, with the study as STUDY_ROOT:
#
#   STUDY_ROOT=outputs/gaussian-softmax DRY_RUN=1 bash run_experiments.sh all
#   STUDY_ROOT=outputs/gaussian-softmax bash run_experiments.sh --list
#
# Generated data, checkpoints, figures, finals, logs and wandb files live directly
# under STUDY_ROOT. Empirical inputs are read from the repository data/. This script:
#
#   study.env   pins DESIGN_GENERATOR; shared_config.sh reads it from STUDY_ROOT,
#               which gives the study its own MC_TAG (suffix _<generator>).
#   a COPY of the Exp-4 aDDM baseline-arm checkpoint: its design is intercept-only, so no
#               generator ever draws a regressor for it, and it does not depend on the
#               CogFormer encoder (e4_probe reuses it when present, re-validates it always).
#
# By default the study regenerates its own shared test set (e1_data) and retrains all 20
# BayesFlow baselines (gap_train), both under the study's generator: benchmark_data and
# the BF trainer share one simulator (ModelFamilyBF), so the baselines train on the
# distribution of the test set they are scored on. The main study's test set and
# baselines are never touched.
#
# --reuse-main-bench-bf instead COPIES the main study's test set and its 20 validated
# baselines (always together: each baseline's npz was scored on that test set). bench5
# designs are intercept-only or all-continuous, where the generators differ only in
# whether a continuous column is rescaled to exactly unit SD, so this saves ~58 GPU-hours
# and keeps the study paired with the main study's datasets, at the price of a test set
# not drawn by the study's own generator.
#
# Copies, never links: a link would let a FORCE=1 stage in the study overwrite the main
# study's files (a hardlinked file rewritten in place changes both).
#
# Idempotent: existing files in the study are never overwritten, nothing is deleted.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USAGE="usage: setup_study.sh <study_dir> <design_generator> [--reuse-main-bench-bf]"
STUDY="${1:?$USAGE}"
[[ "$STUDY" == /* ]] || STUDY="${PROJECT_ROOT}/${STUDY}"
GENERATOR="${2:?$USAGE}"
REUSE_MAIN=0
case "${3:-}" in
    "") ;;
    --reuse-main-bench-bf) REUSE_MAIN=1 ;;
    *) echo "$USAGE" >&2; exit 2 ;;
esac

valid="$(cd "$PROJECT_ROOT" && .venv/bin/python -c \
    'from cogformer.simulator.design_utils import DESIGN_GENERATORS as G; print(" ".join(G))')"
[[ " $valid " == *" $GENERATOR "* ]] || { echo "unknown generator '$GENERATOR' (expected: $valid)" >&2; exit 2; }

mkdir -p "$STUDY"
STUDY="$(cd "$STUDY" && pwd)"
[[ "$STUDY" != "$PROJECT_ROOT" ]] || { echo "the study must not be the repo root" >&2; exit 2; }

if [[ -f "$STUDY/study.env" ]]; then
    have="$(source "$STUDY/study.env"; echo "$DESIGN_GENERATOR")"
    [[ "$have" == "$GENERATOR" ]] || { echo "$STUDY/study.env already pins '$have'" >&2; exit 2; }
else
    cat > "$STUDY/study.env" <<EOF
# Written by experiments/setup_study.sh on $(date +%F) at $(cd "$PROJECT_ROOT" && git rev-parse --short HEAD).
# Sourced by experiments/shared_config.sh when STUDY_ROOT selects this directory.
DESIGN_GENERATOR=$GENERATOR
EOF
fi

MAIN="$PROJECT_ROOT/outputs"
DEST="$STUDY"
copy() {                      # copy <path relative to outputs/> — never overwrites
    local rel="$1"
    [[ -e "$MAIN/$rel" ]] || { echo "  ! missing in main study: outputs/$rel" >&2; return 1; }
    mkdir -p "$(dirname "$DEST/$rel")"
    if [[ -d "$MAIN/$rel" ]]; then
        mkdir -p "$DEST/$rel"
        cp -a --update=none "$MAIN/$rel/." "$DEST/$rel/"
    else
        cp -a --update=none "$MAIN/$rel" "$DEST/$rel"
    fi
    echo "  · outputs/$rel"
}

echo "study ${STUDY}  (generator ${GENERATOR})"
if [[ "$REUSE_MAIN" == 1 ]]; then
    echo "shared benchmark test set (main study's):"
    [[ -e "$MAIN/data/benchmark/seed0/.done" ]] || { echo "  ! main study has no completed test set" >&2; exit 1; }
    copy data/benchmark/seed0
    echo "BayesFlow baselines (main study's 20, validated on that test set):"
    for fam in ddm rdm cdm lba; do
        for c in intercept_only fixed regressed fixed_regressed interaction; do
            copy "checkpoints/model_family/bf/${fam}/${fam}_families_bf_${c}" >/dev/null
            copy "figures/model_family/bf/${fam}/${c}" >/dev/null
            copy "evaluations/model_family/bf/${fam}_families_bf_${c}_evaluations.csv" >/dev/null
            # Last: it is the per-case resume marker, so it lands only after the rest.
            copy "data/model_family/${fam}_${c}_data.npz" >/dev/null
        done
        echo "  · ${fam}: 5 cases"
    done
else
    [[ ! -e "$DEST/data/benchmark/seed0/.done" ]] || echo "  note: the study already has a test set; e1_data will skip"
    echo "test set + BayesFlow baselines: regenerated in the study (e1_data, gap_train)"
fi

echo "Exp-4 aDDM baseline-arm checkpoint:"
copy checkpoints/encoder_rep/bf_baseline/addm_vsvfixed/intercept_only/item0_vsvfixed_e500

echo
echo "next:"
echo "  STUDY_ROOT=${STUDY#$PROJECT_ROOT/} bash run_experiments.sh --list"
echo "  STUDY_ROOT=${STUDY#$PROJECT_ROOT/} DRY_RUN=1 bash run_experiments.sh all"
