#!/usr/bin/env bash
# babysit_bf_cycle2.sh — hand GPU 0 to the rdm BF baseline when ddm finishes.
#
# Cycle 1 (running since 2026-09-02 16:00):
#   ddm -> GPU 0 (cpus 0-23), driver pid 77755
#   cdm -> GPU 1 (cpus 24-47), driver pid 77812
#
# Cycle 2 (this script): ddm is ~1.5 days ahead of cdm, so GPU 0 would sit idle
# until cdm finishes. Wait for the ddm driver to exit, then start rdm on the
# freed GPU 0 with the same CPU partition. cdm on GPU 1 is left untouched.
#
# All five bench cases were simulator-checked (--test) for rdm on 2026-09-04
# before this script was written; run_bf_cycle.sh continues past a failed case
# regardless.
#
# Detached-safe: no dependence on the launching shell. Progress -> stdout.
#
# Usage: setsid nohup bash outputs/babysit_bf_cycle2.sh > outputs/logs/babysit_bf_cycle2.log 2>&1 &
set -uo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2

DDM_DRIVER=77755      # bash outputs/run_bf_cycle.sh ddm 0 0-23
GPU=0
CPUS=0-23

log() { echo "[$(date '+%F %T')] $*"; }

# Match on cmdline, not just liveness: this waits for days, and a bare `kill -0`
# on a recycled pid would block forever after ddm exits.
driver_alive() {
    local cmd
    cmd=$(tr '\0' ' ' < "/proc/$DDM_DRIVER/cmdline" 2>/dev/null) || return 1
    [[ "$cmd" == *run_bf_cycle.sh*ddm* ]]
}

if ! driver_alive; then
    log "ddm driver pid $DDM_DRIVER is not running run_bf_cycle.sh ddm -- refusing to guess; check by hand"
    exit 1
fi

log "waiting for ddm cycle (driver pid $DDM_DRIVER) to finish all cases"
while driver_alive; do sleep 120; done
log "ddm cycle exited; GPU$GPU should be free"

# Safety net: the driver is gone, but a case that outlived it (or a stray run)
# would still hold GPU 0. Clear anything bf_train bound to this GPU.
for p in $(pgrep -f 'experiments.model_family.model_family_bf_train'); do
    if tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$GPU"; then
        log "killing straggler GPU$GPU job pid $p"
        kill -- "-$(ps -o pgid= -p "$p" | tr -d ' ')" 2>/dev/null || kill "$p"
    fi
done
sleep 30

out="outputs/logs/cycle2_rdm_gpu${GPU}.log"
setsid nohup bash outputs/run_bf_cycle.sh rdm "$GPU" "$CPUS" > "$out" 2>&1 &
log "launched rdm cycle pid $! on GPU$GPU (cpus $CPUS) -> $out"
log "babysitter done"
