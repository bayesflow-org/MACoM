"""Resumable random-design evaluation with bounded memory and milestone reports.

Use the Experiment 1 checkpoint/architecture and sampling settings explicitly.
The coordinator interleaves families and shards config indices across GPU workers.
Scores with BayesFlow's public diagnostic functions. Only the three diagnostic
metrics are saved; no plots or draw archives.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import distribution
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from numba import njit

from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from cogformer.simulator.design_utils import (
    build_row_labels, expand_design_config_to_mask, sample_covering_design_configs,
)
from experiments.experiment_1.model_class_cf_ensemble import (
    build_model_class, DEFAULT_FIXED_VALUES, GLOBAL_PARAMS,
)
from experiments.utils import (
    add_interaction_prior_args, add_structure_conditioning_arg,
    build_encoder_input_dim, chunked_sample, infer_free_fixed_intrinsics, seed_case,
)

METRICS = ['NRMSE', 'Posterior Contraction', 'Calibration Error']


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--outdir', required=True, type=Path)
    p.add_argument('--reference_run', type=Path, default=None,
                   help='Replay the stored structural plan from an existing run.json')
    p.add_argument('--reference_inputs', choices=['match', 'regenerate'], default='match',
                   help='Require original input hashes, or explicitly record fresh simulated datasets')
    p.add_argument('--n_configs', type=int, default=1000)
    p.add_argument('--report_every', type=int, default=200)
    p.add_argument('--gpu_ids', default='0,1')
    p.add_argument('--worker_index', type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument('--batch_size', type=int, default=200)
    p.add_argument('--num_obs', type=int, default=500)
    p.add_argument('--num_samples', type=int, default=200)
    p.add_argument('--num_sample_steps', type=int, default=50)
    p.add_argument('--samples_chunk_size', type=int, default=5)
    p.add_argument('--solver', choices=['euler', 'midpoint', 'rk4'], default='rk4')
    p.add_argument('--num_regressors', type=int, default=2)
    p.add_argument('--num_categories', type=int, default=2)
    p.add_argument('--max_num_regressors', type=int, default=4)
    p.add_argument('--max_num_categories', type=int, default=4)
    p.add_argument('--design_generator', choices=['mixture', 'quantile', 'gaussian_softmax'],
                   default='gaussian_softmax')
    p.add_argument('--seed', type=int, default=20261029)
    p.add_argument('--free_prob', type=float, default=0.5)
    p.add_argument('--families', default='DDM,RDM,CDM')
    p.add_argument('--normalize', choices=['range', 'std'], default='range')
    add_interaction_prior_args(p)
    add_structure_conditioning_arg(p)
    for name, default in dict(encoder_num_layers=4, encoder_num_heads=4,
                              encoder_embed_dim=128, decoder_num_layers=8,
                              decoder_num_heads=8, proj_dim=256, num_seeds=32,
                              seed_dim=128, model_embed_dim=8).items():
        p.add_argument('--' + name, type=int, default=default)
    args = p.parse_args()
    for name in ('n_configs', 'report_every', 'batch_size', 'num_obs', 'num_samples',
                 'num_sample_steps', 'samples_chunk_size'):
        if getattr(args, name) <= 0:
            p.error(name + ' must be positive')
    args.checkpoint = args.checkpoint.resolve()
    args.outdir = args.outdir.resolve()
    if args.reference_run is not None:
        args.reference_run = args.reference_run.resolve()
        if args.reference_run.parent == args.outdir:
            p.error('Use a separate output directory when replaying a reference run')
    return args


def build_plan(args):
    if getattr(args, 'reference_run', None) is not None:
        reference = json.loads(args.reference_run.read_text())
        current = settings(args)
        # A shorter prefix is useful for validation. All inference and simulation
        # settings, the checkpoint hash, and RNG seed must still match exactly.
        ignored = {'n_configs', 'report_every', 'schema_version', 'metric_implementation',
                   'reference_run'}
        changed = [k for k, v in reference['settings'].items()
                   if k not in ignored and current.get(k) != v]
        if changed:
            raise ValueError(f'Reference run has different inference settings: {changed}')
        plan = reference['plan'][:args.n_configs]
        if len(plan) != args.n_configs or [p['index'] for p in plan] != list(range(args.n_configs)):
            raise ValueError('Reference plan does not contain the requested configuration prefix')
        return plan
    families, _ = build_model_class(num_workers=1)
    names = args.families.split(',')
    if len(set(names)) != len(names) or any(n not in families for n in names):
        raise ValueError('families must be distinct DDM/RDM/CDM names')
    base, remainder = divmod(args.n_configs, len(names))
    configs = {}
    for j, name in enumerate(names):
        n = base + int(j < remainder)
        if not n:
            configs[name] = []
            continue
        fixed = DEFAULT_FIXED_VALUES[name]
        configs[name] = sample_covering_design_configs(
            n_configs=n, intrinsic_params=families[name].intrinsic_params,
            num_regressors=args.num_regressors,
            free_intrinsics=[p for p in families[name].intrinsic_params if p not in fixed],
            fixed_intrinsics=list(fixed), keep_intercept=True, add_interaction=True,
            free_prob=args.free_prob, global_params=GLOBAL_PARAMS,
            interaction_policy=args.interaction_policy, interaction_prob=args.interaction_prob,
            rng_seed=(args.seed + j) % (2**32),
        )
    # Every prefix is family-balanced, including the first 200 configs.
    return [dict(index=i, family=name, family_index=k, design_config=configs[name][k])
            for i, (k, name) in enumerate(
                (k, name) for k in range(base + bool(remainder)) for name in names
                if k < len(configs[name]))]


def settings(args):
    values = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ('worker_index', 'gpu_ids', 'outdir')}
    with args.checkpoint.open('rb') as checkpoint:
        values['checkpoint_sha256'] = hashlib.file_digest(checkpoint, 'sha256').hexdigest()
    values['sim_backend'] = os.environ.get('COGFORMER_SIM_BACKEND', 'numba')
    values['numba_rng'] = 'per_config_torch_initial_seed'
    values['schema_version'] = 2
    package = distribution('bayesflow')
    direct_url = json.loads(package.read_text('direct_url.json') or '{}')
    values['metric_implementation'] = dict(
        name='bayesflow', version=package.version,
        commit=direct_url.get('vcs_info', {}).get('commit_id'),
        aggregation='median', calibration_resolution=20,
        calibration_min_quantile=0.005, calibration_max_quantile=0.995,
    )
    return values


def bayesflow_metrics():
    # The metric functions operate on NumPy arrays. Using the torch backend avoids
    # initializing a second GPU runtime through Keras/JAX in these PyTorch workers.
    os.environ.setdefault('KERAS_BACKEND', 'torch')
    from bayesflow.diagnostics import metrics
    return metrics


@njit(cache=True)
def seed_numba_rng(seed):
    """Numba's design-generator RNG is separate from NumPy's Python RNG."""
    np.random.seed(seed)


def config_metrics(args, model_class, dc, cats, truth, pred, mask):
    """Score each cell only on datasets where its realized design supports it.

    Sparse categorical interaction cells can be unobserved in individual datasets,
    even under the same structural config. Their masked targets/draws must not enter
    RMSE, prior variance, or coverage. Common cells are vectorized; cells with
    varying support are scored separately with the actual BayesFlow functions.
    """
    mask = np.asarray(mask)
    if mask.ndim == 2:
        mask = np.broadcast_to(mask, truth.shape)
    if mask.shape != truth.shape:
        raise ValueError('Parameter mask must match the truth array')
    common = np.all(mask == 1, axis=0)
    varying = np.any(mask == 1, axis=0) & ~common
    params = model_class.global_param_names
    expanded = expand_design_config_to_mask(
        dc, common, args.max_num_categories, max_num_regressors=args.max_num_regressors)
    row_labels = build_row_labels(expanded, args.max_num_categories,
                                  categorical_regressors=cats, parameter_mask=common)
    bf = bayesflow_metrics()
    records = []

    def score(selected, cell_mask):
        rows, cols = np.nonzero(cell_mask)
        if not len(rows):
            return
        n = int(np.count_nonzero(selected))
        if n < 2:
            logging.warning('Skipping cells with only %s supported datasets', n)
            return
        selected_truth = truth if np.all(selected) else truth[selected]
        selected_pred = pred if np.all(selected) else pred[selected]
        names = [params[c] if r == 0 else f'{params[c]} -- {row_labels[r]}'
                 for r, c in zip(rows, cols)]
        kwargs = dict(estimates=selected_pred[:, :, rows, cols],
                      targets=selected_truth[:, rows, cols], variable_names=names,
                      aggregation=np.median)
        results = [
            bf.root_mean_squared_error(**kwargs, normalize=args.normalize),
            bf.posterior_contraction(**kwargs),
            bf.calibration_error(**kwargs, resolution=20, min_quantile=0.005, max_quantile=0.995),
        ]
        for i, (parameter, c) in enumerate(zip(names, cols)):
            values = {result['metric_name']: float(result['values'][i]) for result in results}
            if not all(np.isfinite(v) for v in values.values()):
                raise ValueError(f'BayesFlow returned nonfinite metrics for {parameter}')
            records.append(dict(parameter=parameter, scope='pi' if params[c] in GLOBAL_PARAMS else 'local',
                                n_datasets=n, **values))

    score(np.ones(len(truth), dtype=bool), common)
    for r, c in np.argwhere(varying):
        one_cell = np.zeros_like(common)
        one_cell[r, c] = True
        score(mask[:, r, c] == 1, one_cell)
    return records


def result_path(outdir, index):
    return outdir / 'configs' / f'{index:04d}.json'


def report(outdir, count):
    """Equal weight per config for headline means; per-cell tables retain support counts."""
    results = [json.loads(result_path(outdir, i).read_text()) for i in range(count)]
    run_path = outdir / 'run.json'
    run_settings = json.loads(run_path.read_text())['settings'] if run_path.exists() else {}
    batch_size = run_settings.get('batch_size')
    implementation = run_settings.get('metric_implementation')
    if implementation is not None and any(r.get('metric_implementation') != implementation for r in results):
        raise ValueError('Configuration records have a different metric implementation from run.json')
    cells = pd.DataFrame([dict(config_index=r['index'], family=r['family'],
                               **{'n_datasets': batch_size, **cell})
                          for r in results for cell in r['metrics']])
    by_config = cells.groupby(['config_index', 'family', 'scope'])[METRICS].mean().reset_index()
    rows = []
    for family in [*dict.fromkeys(r['family'] for r in results), 'ALL']:
        subset = by_config if family == 'ALL' else by_config[by_config.family == family]
        for scope, group in subset.groupby('scope'):
            row = dict(family=family, scope=scope, n_configs=len(group))
            for metric in METRICS:
                row[metric] = float(group[metric].mean())
                row[metric + ' median'] = float(group[metric].median())
                row[metric + ' SD'] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
                row[metric + ' SE'] = row[metric + ' SD'] / len(group)**0.5
            rows.append(row)
    summaries = outdir / 'reports'
    summaries.mkdir(exist_ok=True)
    stem = f'metrics_{count:04d}'
    pd.DataFrame(rows).to_csv(summaries / f'{stem}.csv', index=False)
    cells.to_csv(summaries / f'{stem}_per_config_cells.csv', index=False)
    cells.groupby(['family', 'scope', 'parameter'])[METRICS].agg(
        ['mean', 'std', 'count']).to_csv(summaries / f'{stem}_by_parameter.csv')
    lines = [f'# Ensemble metrics after {count} configurations', '',
             (f"Metrics computed by BayesFlow {implementation['version']} using its public "
              'root_mean_squared_error, posterior_contraction, and calibration_error functions.'
              if implementation else 'Metric implementation: legacy/unrecorded.'), '',
             'Means give equal weight to each configuration; within a configuration, '
             'active coefficient cells have equal weight. Local coefficients and global pi '
             'are reported separately. NRMSE uses '
             f"{run_settings.get('normalize', 'range')} normalization. "
             'SD and SE in the CSV describe variation between '
             'config-level averages, not uncertainty within an individual dataset.', '',
             '| Family | Parameters | Configs | NRMSE | Contraction | Calibration error |',
             '|---|---|---:|---:|---:|---:|']
    if run_settings.get('reference_run') and run_settings.get('reference_inputs') == 'regenerate':
        lines[4:4] = ['Freshly simulated datasets for the reference structural plan; original datasets '
                      'and posterior draws were not retained. NumPy, PyTorch, and Numba are seeded '
                      'per configuration.', '']
    for row in rows:
        lines.append(f"| {row['family']} | {row['scope']} | {row['n_configs']} | "
                     + ' | '.join(f'{row[m]:.5f}' for m in METRICS) + ' |')
    (summaries / f'{stem}.md').write_text('\n'.join(lines) + '\n')
    atomic_json(summaries / f'{stem}.json', dict(created_at=now(), n_configs=count,
                                               metric_implementation=implementation,
                                               reference_run=run_settings.get('reference_run'),
                                               reference_inputs=run_settings.get('reference_inputs'),
                                               summary=rows))
    logging.info('REPORT %s\n%s', count, '\n'.join(lines))
    # Written last: workers use this as the milestone barrier.
    atomic_json(outdir / 'latest_report.json', dict(n_configs=count, report=str(summaries / f'{stem}.md')))
    return rows


def load_model(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this sweep; refusing an unintended CPU run')
    architecture = {k: getattr(args, k) for k in (
        'encoder_num_layers', 'encoder_num_heads', 'encoder_embed_dim', 'decoder_num_layers',
        'decoder_num_heads', 'proj_dim', 'num_seeds', 'seed_dim', 'model_embed_dim')}
    model = CogFormer(
        encoder_input_dim=build_encoder_input_dim(args.max_num_regressors, args.max_num_categories, True),
        encoder_structure_conditioning=args.encoder_structure_conditioning,
        num_models=3, decoder_layer_design='mixed_attention',
        decoder_layer_kwargs={'mab_first': True}, self_attn_padding_mask=True, **architecture,
    ).to('cuda')
    state = torch.load(args.checkpoint, map_location='cuda')
    if 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in state.items()}, strict=True)
    logging.info('Loaded checkpoint strictly on %s', torch.cuda.get_device_name(0))
    return torch.compile(model.eval(), dynamic=True)


@torch.no_grad()
def worker(args):
    run = json.loads((args.outdir / 'run.json').read_text())
    if run['settings'] != settings(args):
        raise ValueError('Worker settings differ from the recorded run')
    plan = run['plan']
    n_workers = len(args.gpu_ids.split(','))
    bayesflow_metrics()  # Import before per-config seeding, preserving the sampling RNG stream.
    families, model_class = build_model_class(num_workers=1)
    model = load_model(args)
    adapter = Adapter()
    for item in plan:
        index, name, dc = item['index'], item['family'], item['design_config']
        if index % n_workers != args.worker_index:
            continue
        # Complete the same prefix before either worker starts the next block.
        barrier = index // args.report_every * args.report_every
        while barrier and (not (args.outdir / 'latest_report.json').exists() or
                           json.loads((args.outdir / 'latest_report.json').read_text())['n_configs'] < barrier):
            time.sleep(2)
        path = result_path(args.outdir, index)
        if path.exists():
            continue
        seed_case(args.seed, 'ensemble_metrics_sweep', index, name)
        seed_numba_rng(torch.initial_seed() % (2**32))
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        mf = families[name]
        free, fixed, values = infer_free_fixed_intrinsics(dc, mf.intrinsic_params, DEFAULT_FIXED_VALUES[name])
        logging.info('Config %s/%s %s simulation', index + 1, args.n_configs, name)
        samples = mf.batch_sample(
            design_config=dc, batch_size=args.batch_size, num_obs=args.num_obs,
            mask_randomizer_kwargs=dict(free_intrinsics=free, fixed_intrinsics=fixed, fixed_values=values),
            max_num_regressors=args.max_num_regressors, max_num_categories=args.num_categories,
            min_num_categories=args.num_categories, keep_intercept=True, add_interaction=True,
            flatten_param_outputs=True, fixed_config=True, link_fun=model_class.link_funs[name],
            pad_to_max_num_categories=args.max_num_categories, design_generator=args.design_generator,
        )
        samples['model_ids'] = np.full(args.batch_size, model_class.model_index[name], dtype=np.int64)
        samples['param_matrices'], samples['param_masks'] = model_class.lift_to_global_space(
            name, samples['param_matrices'], samples['param_masks'])
        samples['n_params'] = model_class.n_global_params
        adapted = adapter.adapt(samples, device='cuda')
        n_params = model_class.n_global_params
        truth = adapted['param_matrices'].cpu().numpy().reshape(args.batch_size, -1, n_params)
        mask = adapted['param_masks'].cpu().numpy().reshape(args.batch_size, -1, n_params)
        data_hash = hashlib.sha256()
        for array in (adapted['input_data'].cpu().numpy(), truth, mask):
            data_hash.update(np.ascontiguousarray(array).tobytes())
        input_sha256 = data_hash.hexdigest()
        reference_info = {}
        if args.reference_run is not None:
            reference = json.loads(result_path(args.reference_run.parent, index).read_text())
            matches = reference['input_sha256'] == input_sha256
            if not matches and args.reference_inputs == 'match':
                raise ValueError(f'Config {index}: regenerated inputs differ from reference run')
            reference_info = dict(reference_input_sha256=reference['input_sha256'],
                                  reference_inputs_match=matches)
        pred = chunked_sample(model, adapted, args.num_samples, args.num_sample_steps,
                              args.solver, args.samples_chunk_size)
        pred = pred.reshape(args.batch_size, args.num_samples, truth.shape[1], n_params)
        cats = [f'u_{j+1}' for j, v in enumerate(samples['discrete_masks'][0][:args.num_regressors]) if v == 1]
        records = config_metrics(args, model_class, dc, cats, truth, pred, mask)
        payload = dict(**item, completed_at=now(), elapsed_seconds=time.monotonic()-start,
                       input_sha256=input_sha256, metrics=records, **reference_info,
                       metric_implementation=run['settings']['metric_implementation'],
                       gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        atomic_json(path, payload)
        logging.info('Saved config %s: %.1fs, GPU peak %.2f GiB', index + 1,
                     payload['elapsed_seconds'], payload['gpu_peak_allocated_bytes']/1024**3)
        del samples, adapted, pred, truth, mask


def coordinate(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Leave CUDA_VISIBLE_DEVICES unset; use --gpu_ids')
    gpu_ids = args.gpu_ids.split(',')
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError('GPU IDs must be distinct')
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / 'configs').mkdir(exist_ok=True)
    with (args.outdir / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_path = args.outdir / 'run.json'
        expected = settings(args)
        if run_path.exists():
            if json.loads(run_path.read_text())['settings'] != expected:
                raise ValueError('Existing run has different settings; use another output directory')
        else:
            atomic_json(run_path, dict(created_at=now(), settings=expected, plan=build_plan(args)))
        cpus = sorted(os.sched_getaffinity(0))
        processes, handles = [], []
        started = time.monotonic()
        try:
            for j, gpu in enumerate(gpu_ids):
                affinity = cpus[j*len(cpus)//len(gpu_ids):(j+1)*len(cpus)//len(gpu_ids)]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2',
                           MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMBA_NUM_THREADS='2')
                log = (args.outdir / f'worker_{j}.log').open('a')
                handles.append(log)
                cmd = ['taskset', '-c', ','.join(map(str, affinity)), sys.executable,
                       '-m', 'experiments.experiment_1.ensemble_metrics_sweep',
                       *sys.argv[1:], '--worker_index', str(j)]
                processes.append(subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL,
                                                   stdout=log, stderr=subprocess.STDOUT,
                                                   start_new_session=True))
                logging.info('Worker %s GPU %s PID %s', j, gpu, processes[-1].pid)
            milestones = sorted(set([*range(args.report_every, args.n_configs+1, args.report_every), args.n_configs]))
            while True:
                completed = {int(p.stem) for p in (args.outdir / 'configs').glob('*.json')}
                for milestone in milestones:
                    if all(i in completed for i in range(milestone)) and not (
                            args.outdir / 'reports' / f'metrics_{milestone:04d}.json').exists():
                        report(args.outdir, milestone)
                failures = [p.returncode for p in processes if p.poll() not in (None, 0)]
                if failures:
                    raise RuntimeError(f'Worker failed: {failures}; see worker logs')
                done = len(completed) == args.n_configs
                atomic_json(args.outdir / 'progress.json', dict(
                    updated_at=now(), status='complete' if done else 'running',
                    completed_configs=len(completed), target_configs=args.n_configs,
                    elapsed_seconds=time.monotonic()-started, coordinator_pid=os.getpid(),
                    workers=[dict(pid=p.pid, returncode=p.poll()) for p in processes]))
                if done:
                    for proc in processes:
                        if proc.wait(timeout=30) != 0:
                            raise RuntimeError('Worker failed after its final output')
                    return
                if all(p.poll() is not None for p in processes):
                    raise RuntimeError('Workers exited before all configurations completed')
                time.sleep(5)
        except BaseException as error:
            for proc in processes:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait()
            atomic_json(args.outdir / 'progress.json', dict(updated_at=now(), status='failed', error=str(error)))
            raise
        finally:
            for handle in handles:
                handle.close()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args()
    if args.worker_index is None:
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Received signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        coordinate(args)
    else:
        worker(args)


if __name__ == '__main__':
    main()
