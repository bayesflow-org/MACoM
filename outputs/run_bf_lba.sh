#!/usr/bin/env bash
# run_bf_lba.sh — LBA BayesFlow baseline, with the one environment fix LBA needs.
#
# NUMBA_THREADING_LAYER=workqueue is LOAD-BEARING. model_family_bf_train builds a
# keras OnlineDataset with use_multiprocessing=True, which FORKS `--sim_workers`
# child processes. numba's default threading layer here is OpenMP-backed, and the
# parent has already initialised libgomp by the time the fork happens, so every
# child dies instantly with
#
#     Terminating: fork() called from a process already using GNU OpenMP, this is unsafe.
#
# while the parent waits on a futex forever: 0% CPU, GPU allocated but idle, an
# empty log, and NO exit. run_bf_cycle.sh cannot detect this -- it only checks exit
# codes, and a deadlocked case never exits. That cost 21 hours on 2026-09-08/09.
# `workqueue` is fork-safe and costs nothing here, because the design is already
# 20 worker processes x 1 numba thread; intra-process numba threading is unused.
#
# Measured 2026-09-09: default layer -> deadlock (killed at 300s, 0 epochs);
# workqueue -> intercept_only 1 epoch / 2 steps in 33.8s, exit 0.
#
# This is NOT LBA-specific in principle -- DDM/CDM/RDM run the same code path and
# happen not to trip it -- so if an RDM case ever hangs with an empty log, this is
# the first thing to try. It lives in a wrapper rather than in run_bf_cycle.sh
# because that file was being executed by the live RDM driver when this was
# written, and bash reads scripts lazily by byte offset: editing a running script
# corrupts its execution.
#
# Usage: setsid nohup bash outputs/run_bf_lba.sh > outputs/logs/cycle3_lba_gpu1.log 2>&1 &
set -uo pipefail
cd /home/jerry/Documents/research/projects/CogFormer-2

export NUMBA_THREADING_LAYER=workqueue
export MPLBACKEND=Agg

exec bash outputs/run_bf_cycle.sh lba 1 24-47
