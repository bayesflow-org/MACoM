"""Dependency scheduler behind `run_experiments.sh remaining_two_gpu`.

Every task still runs through the canonical launcher, with one process per GPU.
The existing Experiment 1 checkpoint and benchmark files must already be complete.
"""
from experiments.paths import RESULTS_ROOT
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


# Order is priority among ready tasks, not execution order. Start the longest
# checkpoint dependency and the native BF comparison on separate cards first.
TASKS = {
    'e3_finetune': (),
    'e1a_bf': (),
    'e3_zeroshot': (),
    'e3_ft_validate': ('e3_finetune',),
    'e2_apply': (),
    'e3_bf': (),
    'e4_probe': (),
    'e1a_c2st': ('e1a_bf',),
    'e1a_finalize': ('e1a_c2st',),
    'e2_refit_svfix': ('e2_apply',),
    'e2_validate_ddm': ('e2_refit_svfix',),
    'e2_validate_rdm': ('e2_apply',),
    'e2_validate_cdm': ('e2_apply',),
    'e2_ppc': ('e2_refit_svfix',),
    'e2_finalize': ('e2_validate_ddm', 'e2_validate_rdm', 'e2_validate_cdm', 'e2_ppc'),
    'e3_original': (),
    'e3_paired_c2st': ('e3_ft_validate', 'e3_original'),
    'e3_compare': ('e3_zeroshot', 'e3_ft_validate'),
    'e3_ppc_apply': ('e3_finetune',),
    'e3_ppc': ('e3_ppc_apply',),
    'e3_realdata_validate': ('e3_finetune',),
    'e3_finalize': ('e3_paired_c2st', 'e3_compare', 'e3_ppc', 'e3_realdata_validate'),
    'e3_gap_c2st': ('e3_bf', 'e3_ft_validate'),
    'e3_gap_finalize': ('e3_finalize', 'e3_gap_c2st'),
    'e4_compare': ('e4_probe',),
    'e4_representation': ('e4_compare', 'e3_finetune'),
    'e4_finalize': ('e4_representation',),
}

CPU_TASKS = {'e1a_c2st', 'e1a_finalize', 'e2_finalize', 'e3_original',
             'e3_paired_c2st', 'e3_compare', 'e3_finalize', 'e3_gap_c2st',
             'e3_gap_finalize', 'e4_finalize'}


def ready_tasks(status, tasks=None):
    tasks = TASKS if tasks is None else tasks
    return [name for name, deps in tasks.items()
            if status[name]['status'] == 'pending'
            and all(status[d]['status'] == 'complete' for d in deps)]


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def verify_bf_inputs(families, root=(RESULTS_ROOT)):
    """Reject parameter-only pairing: every saved observation must match too."""
    import numpy as np
    checked = []
    for family in families:
        for case in ('intercept_only', 'fixed', 'regressed', 'fixed_regressed', 'interaction'):
            filename = f'{family}_{case}_data.npz'
            with np.load(root / 'data/benchmark/seed0' / filename) as benchmark, \
                    np.load(root / 'data/model_family' / filename) as result:
                for key in benchmark.files:
                    np.testing.assert_array_equal(result[key], benchmark[key],
                                                  err_msg=f'{filename}: {key} differs')
            checked.append(filename)
    return checked


def save_status(path, status, gpu_ids):
    payload = dict(updated_at=timestamp(), pid=os.getpid(), gpu_ids=gpu_ids,
                   tasks=status,
                   status=('complete' if all(v['status'] == 'complete' for v in status.values())
                           else 'failed' if all(v['status'] in ('complete', 'failed', 'blocked')
                                                for v in status.values()) else 'running'))
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launcher', required=True, type=Path)
    parser.add_argument('--gpu0', default='0')
    parser.add_argument('--gpu1', default='1')
    args = parser.parse_args()
    gpu_ids = [args.gpu0, args.gpu1]
    if len(set(gpu_ids)) != 2:
        parser.error('Two distinct physical GPU IDs are required')
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        parser.error('Leave CUDA_VISIBLE_DEVICES unset for the scheduler; use GPU0/GPU1')
    if os.environ.get('BF_REUSE_CHECKPOINTS') != '1':
        parser.error('This continuation requires BF_REUSE_CHECKPOINTS=1 and copied BF weights')
    run_dir = Path(os.environ.get('CONTINUATION_DIR', RESULTS_ROOT / 'continuation'))
    run_dir.mkdir(parents=True, exist_ok=True)
    # Prevent an accidental second scheduler from assigning the same GPUs/study.
    import fcntl
    lock = (run_dir / 'scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = run_dir / 'status.json'
    status = {name: dict(status='pending', dependencies=list(deps)) for name, deps in TASKS.items()}
    previous = json.loads(status_path.read_text())['tasks'] if status_path.exists() else {}
    for name, record in previous.items():
        if name in status and record['status'] == 'complete':
            status[name] = record
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 2:
        parser.error('At least two CPU cores are required')
    halves = [cpus[:len(cpus)//2], cpus[len(cpus)//2:]]
    active = {}
    try:
        while True:
            for gpu, (proc, name, log) in list(active.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                log.close()
                if rc == 0 and name in ('e1a_bf', 'e3_bf'):
                    families = ('ddm', 'rdm', 'cdm') if name == 'e1a_bf' else ('lba',)
                    try:
                        status[name]['paired_inputs_verified'] = verify_bf_inputs(families)
                    except Exception as error:
                        rc = 1
                        status[name]['verification_error'] = str(error)
                status[name].update(status='complete' if rc == 0 else 'failed',
                                    returncode=rc, finished_at=timestamp())
                print(f'{timestamp()} GPU {gpu}: {name} exited {rc}', flush=True)
                del active[gpu]
            # Propagate failures, while allowing unrelated experiments to finish.
            for name, deps in TASKS.items():
                if status[name]['status'] == 'pending' and any(
                        status[d]['status'] in ('failed', 'blocked') for d in deps):
                    status[name].update(status='blocked', reason='upstream task failed')
            for slot, gpu in enumerate([*gpu_ids, 'cpu']):
                ready = [name for name in ready_tasks(status)
                         if (name in CPU_TASKS) == (gpu == 'cpu')]
                if gpu in active or not ready:
                    continue
                name = ready[0]
                stage = 'e2_validate' if name.startswith('e2_validate_') else name
                affinity = cpus if gpu == 'cpu' else halves[slot]
                workers = max(1, min(8, len(cpus)//3)) if gpu == 'cpu' else max(1, min(12, len(affinity)-2))
                threads = '1' if gpu == 'cpu' else '2'
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu == 'cpu' else gpu,
                           NUM_GPU_JOBS='2', NPROC='1',
                           PIPELINE_LABEL=f'gpu{gpu}_{name}',
                           C2ST_WORKERS=str(workers), OMP_NUM_THREADS=threads,
                           MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
                           NUMEXPR_NUM_THREADS=threads)
                if name.startswith('e2_validate_'):
                    # The canonical stage's marker is CDM's output; independent
                    # subsets must run even if CDM happened to finish first.
                    env.update(E2_FAMILIES=name.removeprefix('e2_validate_'), FORCE='1')
                # A failed evaluation may have created its coarse directory marker.
                # Re-run that stage rather than accepting a partially written output.
                if previous.get(name, {}).get('status') in ('failed', 'running') and name != 'e3_finetune':
                    env['FORCE'] = '1'
                log_path = run_dir / f'{name}.log'
                log = log_path.open('a')
                command = ['taskset', '-c', ','.join(map(str, affinity)),
                           'bash', str(args.launcher), stage]
                proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, start_new_session=True)
                active[gpu] = (proc, name, log)
                status[name].update(status='running', gpu=gpu, pid=proc.pid,
                                    started_at=timestamp(), log=str(log_path), command=command,
                                    cpu_affinity=affinity)
                print(f'{timestamp()} GPU {gpu}: starting {name} (pid {proc.pid})', flush=True)
            save_status(status_path, status, gpu_ids)
            if not active:
                break
            time.sleep(10)
    except BaseException:
        # Children must not survive a failed coordinator and later collide on GPUs.
        for proc, _, log in active.values():
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait()
            log.close()
        raise
    finally:
        lock.close()
    failed = [name for name, v in status.items() if v['status'] != 'complete']
    if failed:
        raise SystemExit('Incomplete stages: ' + ', '.join(failed))
    print('All remaining experiments complete.', flush=True)


if __name__ == '__main__':
    main()
