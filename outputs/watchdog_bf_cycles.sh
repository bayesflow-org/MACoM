#!/usr/bin/env bash
# watchdog_bf_cycles.sh — detect and recover the silent BF-training deadlock.
#
# THE FAILURE IT EXISTS FOR. model_family_bf_train forks its sim workers (keras
# OnlineDataset, use_multiprocessing=True). With numba's default OpenMP-backed
# threading layer the children die instantly --
#   "Terminating: fork() called from a process already using GNU OpenMP"
# -- and the parent waits on a futex FOREVER: 0% CPU, GPU memory held but idle, no
# exit code. run_bf_cycle.sh only inspects exit codes, so a deadlocked case never
# fails and never advances; the cycle just stops. This burned 21 hours on LBA
# (2026-09-08/09) before anyone looked. See outputs/run_bf_lba.sh for the fix,
# NUMBA_THREADING_LAYER=workqueue, which this script applies on restart.
#
# WHY IT RESTARTS THE CYCLE RATHER THAN THE CASE. Killing just the hung python
# makes the driver's `taskset ... python` return non-zero, at which point
# run_bf_cycle.sh logs FAILED and moves to the NEXT case -- the stuck case would be
# skipped silently, which is the same data loss in slower motion. So a stall kills
# the whole cycle for that model and relaunches every case that has no checkpoint
# yet, via run_bf_retrain.sh (which takes an explicit case list).
#
# WHAT COUNTS AS DONE. The driver's own "[model] done case=X" line, which is
# written only after python exits 0. NOT the presence of a checkpoint: BayesFlow's
# BasicWorkflow saves checkpoint_filepath EVERY EPOCH, so
# .../rdm_families_bf_regressed/model.keras exists and is seconds old while that
# case is still at epoch 212/2000. Treating a checkpoint as completion would make
# this watchdog silently drop in-flight cases on restart -- exactly the data loss
# it exists to prevent. Driver logs are scanned by glob so the record survives
# restarts, which write to their own cycle3_*_restartN.log.
#
# STALENESS. A healthy run appends progress-bar output continuously -- the case
# logs reach ~38 MB. STALE_SEC is deliberately generous (default 1800s) because
# JIT warm-up before epoch 1 is silent for a few minutes; a 21-hour hang is still
# caught within half an hour.
#
# Usage: setsid nohup bash outputs/watchdog_bf_cycles.sh > outputs/logs/watchdog_bf_cycles.log 2>&1 &
set -uo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2

STALE_SEC="${STALE_SEC:-1800}"
POLL_SEC="${POLL_SEC:-120}"
MAX_RESTARTS="${MAX_RESTARTS:-2}"
CASES=(intercept_only fixed regressed fixed_regressed interaction)
SELF=$$

declare -A GPU=( [rdm]=0 [lba]=1 )
declare -A CPUS=( [rdm]=0-23 [lba]=24-47 )
declare -A RESTARTS=( [rdm]=0 [lba]=0 )

log() { echo "[$(date '+%F %T')] $*"; }

# Match either driver script for this model. Never match this watchdog itself.
driver_pids() {
    local m=$1
    pgrep -f "run_bf_(cycle|retrain)\.sh ${m} " 2>/dev/null | grep -v "^${SELF}$"
}

trainer_pids() {
    local m=$1
    pgrep -f "model_family_bf_train --model ${m}( |\$)" 2>/dev/null | grep -v "^${SELF}$"
}

done_cases() {
    local m=$1
    grep -ho "\[${m}\] done case=[a-z_]*" \
        outputs/logs/cycle*.log outputs/logs/retrain*.log 2>/dev/null \
        | sed 's/.*done case=//' | sort -u
}

remaining_cases() {
    local m=$1 c d found
    for c in "${CASES[@]}"; do
        found=""
        for d in $(done_cases "$m"); do [ "$c" = "$d" ] && found=1; done
        [ -z "$found" ] && echo "$c"
    done
}

# Seconds since the newest case log for this model was written. Empty if none yet.
log_age() {
    local m=$1 f
    f=$(ls -t outputs/logs/bf_${m}_*.log 2>/dev/null | head -1)
    [ -z "$f" ] && return 1
    echo $(( $(date +%s) - $(stat -c %Y "$f") ))
}

restart_model() {
    local m=$1 rem pids
    rem=$(remaining_cases "$m" | tr '\n' ' ')
    if [ -z "${rem// /}" ]; then
        log "[$m] stalled but every case has a checkpoint; not restarting"
        return 0
    fi

    # DRY_RUN exercises detection + case bookkeeping + the exact relaunch argv
    # without touching a live run. Guard is first, before any kill.
    if [ -n "${DRY_RUN:-}" ]; then
        log "[$m] DRY_RUN: would kill driver=[$(driver_pids "$m" | tr '\n' ' ')] trainers=[$(trainer_pids "$m" | tr '\n' ' ')]"
        log "[$m] DRY_RUN: would run -> NUMBA_THREADING_LAYER=workqueue MPLBACKEND=Agg bash outputs/run_bf_retrain.sh $m ${GPU[$m]} ${CPUS[$m]} $rem"
        return 0
    fi

    log "[$m] killing stalled cycle (driver: $(driver_pids "$m" | tr '\n' ' ') trainers: $(trainer_pids "$m" | tr '\n' ' '))"
    # Drivers first so the loop cannot start the next case while we clean up.
    pids=$(driver_pids "$m"); [ -n "$pids" ] && kill -9 $pids 2>/dev/null
    sleep 2
    pids=$(trainer_pids "$m"); [ -n "$pids" ] && kill -9 $pids 2>/dev/null
    sleep 20

    pids=$(trainer_pids "$m")
    [ -n "$pids" ] && { log "[$m] WARNING: survivors after kill -9: $pids"; }

    RESTARTS[$m]=$(( ${RESTARTS[$m]} + 1 ))
    local out="outputs/logs/cycle3_${m}_gpu${GPU[$m]}_restart${RESTARTS[$m]}.log"
    log "[$m] relaunching cases: $rem -> $out (restart ${RESTARTS[$m]}/${MAX_RESTARTS})"
    # The fix. run_bf_retrain.sh reaches python through `taskset ... env ...`, and
    # env ADDS to the inherited environment, so exported vars propagate.
    NUMBA_THREADING_LAYER=workqueue MPLBACKEND=Agg \
        setsid nohup bash outputs/run_bf_retrain.sh "$m" "${GPU[$m]}" "${CPUS[$m]}" $rem \
        > "$out" 2>&1 &
    log "[$m] relaunched as pid $!"
}

log "watchdog up: stale>${STALE_SEC}s, poll ${POLL_SEC}s, max ${MAX_RESTARTS} restarts/model"
for m in rdm lba; do log "[$m] done so far: $(done_cases "$m" | tr '\n' ' ')"; done

while true; do
    active=0
    for m in rdm lba; do
        [ -z "$(driver_pids "$m")" ] && continue
        active=1

        age=$(log_age "$m") || { log "[$m] driver alive, no case log yet"; continue; }
        if [ "$age" -le "$STALE_SEC" ]; then continue; fi

        log "[$m] STALL: newest case log idle ${age}s (>${STALE_SEC}s)"
        f=$(ls -t outputs/logs/bf_${m}_*.log 2>/dev/null | head -1)
        if grep -q "already using GNU OpenMP" "$f" 2>/dev/null; then
            log "[$m] confirmed: fork/OpenMP deadlock signature in $f"
        else
            log "[$m] no fork signature in $f -- stalled for another reason; restarting anyway"
        fi

        if [ "${RESTARTS[$m]}" -ge "$MAX_RESTARTS" ]; then
            log "[$m] already restarted ${RESTARTS[$m]} time(s); NOT restarting again. Needs a human."
            continue
        fi
        restart_model "$m"
    done

    [ "$active" -eq 0 ] && { log "no BF drivers running; watchdog exiting"; break; }
    sleep "$POLL_SEC"
done
