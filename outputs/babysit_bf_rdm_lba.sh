#!/usr/bin/env bash
# babysit_bf_rdm_lba.sh — hand both GPUs to the RDM and LBA BayesFlow baselines
# once the DDM/CDM amortization-gap run finishes.
#
#   rdm -> GPU 0 (cpus 0-23)      lba -> GPU 1 (cpus 24-47)
#
# Both go through outputs/run_bf_cycle.sh, unchanged, so they get exactly the
# settings that produced the DDM/CDM baselines: bench5 cases, 2000 epochs,
# batch 256, 100 steps/epoch, 20 sim workers per 24-CPU partition.
#
# WHY LBA CAN RUN AT ALL. model_family_bf_train had no LBA entry until
# LBA_BF_CASE_CONFIGS was derived from RDM's (experiments/configs.py). And until
# 2026-09-08 it built the simulator as `model_cls()`, i.e. LBA(fixed_A=None) --
# the legacy parameterization -- while every other LBA arm pins fixed_A=0.5.
# `a` MEANS a different thing under the two, so the baseline would have been
# incomparable to the finetuned arm it exists to be compared against, and nothing
# downstream could have detected it: same shapes, same parameter names. LBA_CONFIG
# now carries model_kwargs={"fixed_A": 0.5} and both bf_train and bf_validate read
# it. Verify before trusting a run:
#     grep -c 'fixed_A' experiments/configs.py
#
# WHY THE WAIT IS A pgrep LOOP, NOT A PID. babysit_bf_cycle2.sh matched one
# hardcoded driver pid and exited 1 when that pid was not running the expected
# script. Its log never appeared and the rdm cycle it was meant to start never
# ran. Polling for "no process matches the driver command" has no such failure
# mode, and an ALREADY-finished run is treated as success rather than an error.
#
# Detached-safe: no dependence on the launching shell. Progress -> stdout.
#
# Usage: setsid nohup bash outputs/babysit_bf_rdm_lba.sh > outputs/logs/babysit_bf_rdm_lba.log 2>&1 &
set -uo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2

# Harmless for training (model_family_bf_train imports no matplotlib) but set
# anyway: DISPLAY=:0 is stale under this Wayland session, and anything that does
# import matplotlib parks in unix_wait_for_peer forever instead of erroring.
export MPLBACKEND=Agg

DRIVER_PAT='bash outputs/run_amortization_gap.sh'
GAP_LOG=outputs/logs/amortization_gap.log

log() { echo "[$(date '+%F %T')] $*"; }

if pgrep -f "$DRIVER_PAT" > /dev/null; then
    log "waiting for the DDM/CDM amortization-gap run to finish"
    while pgrep -f "$DRIVER_PAT" > /dev/null; do sleep 120; done
    log "amortization-gap driver exited"
else
    log "amortization-gap driver is not running; treating it as already finished"
fi

# Report, do not gate: RDM/LBA training does not depend on whether the DDM/CDM
# comparison succeeded, and refusing to start three days of GPU work over a
# missing summary CSV would waste the window.
if grep -q "all stages complete" "$GAP_LOG" 2>/dev/null; then
    log "gap run reached 'all stages complete'"
else
    log "WARNING: '$GAP_LOG' has no 'all stages complete' line -- the DDM/CDM"
    log "WARNING: comparison may be incomplete. Starting RDM/LBA anyway."
fi
for m in ddm cdm; do
    s="./outputs/figures/c2st/${m}/${m}_c2st_summary.csv"
    [ -f "$s" ] && log "found $s" || log "WARNING: missing $s"
done

# Safety net: the driver is gone, but a case that outlived it would still hold a
# card. Clear anything bf_validate/cf_validate left bound to either GPU.
for p in $(pgrep -f 'model_family_bf_validate|model_class_cf_validate'); do
    log "killing straggler pid $p"
    kill -- "-$(ps -o pgid= -p "$p" | tr -d ' ')" 2>/dev/null || kill "$p"
done
sleep 30

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | while read -r l; do log "gpu $l"; done

out_rdm="outputs/logs/cycle3_rdm_gpu0.log"
out_lba="outputs/logs/cycle3_lba_gpu1.log"
setsid nohup bash outputs/run_bf_cycle.sh rdm 0 0-23  > "$out_rdm" 2>&1 &
log "launched rdm cycle pid $! on GPU0 (cpus 0-23) -> $out_rdm"
setsid nohup bash outputs/run_bf_cycle.sh lba 1 24-47 > "$out_lba" 2>&1 &
log "launched lba cycle pid $! on GPU1 (cpus 24-47) -> $out_lba"
log "babysitter done; expect ~2.5-3.5 days per model, 5 cases each"
