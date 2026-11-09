"""Check milestone interpretation and the metric-only path against known answers."""
import contextlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.experiment_1 import ensemble_metrics_sweep as sweep


class MetricsSweepTests(unittest.TestCase):
    def test_numba_rng_reset_repeats_actual_correlation_draws(self):
        from cogformer.utils.simulator_utils import sample_lkj_correlation
        sweep.seed_numba_rng(45)
        first = sample_lkj_correlation(3, 1.5)
        second = sample_lkj_correlation(3, 1.5)
        self.assertFalse(np.array_equal(first, second))
        sweep.seed_numba_rng(45)
        np.testing.assert_array_equal(first, sample_lkj_correlation(3, 1.5))

    def test_balanced_prefix_and_reproducible_structural_plan(self):
        args = SimpleNamespace(n_configs=1000, families='DDM,RDM,CDM', seed=13,
                               num_regressors=2, free_prob=.5,
                               interaction_policy='optional', interaction_prob=None)
        plan = sweep.build_plan(args)
        self.assertEqual(plan, sweep.build_plan(args))
        self.assertEqual(len(plan), 1000)
        self.assertEqual([sum(p['family'] == f for p in plan[:200]) for f in ['DDM', 'RDM', 'CDM']],
                         [67, 67, 66])
        for worker in range(2):
            self.assertEqual(sum(p['index'] % 2 == worker for p in plan[:200]), 100)
        self.assertEqual([p['index'] for p in plan], list(range(1000)))

    def test_metrics_known_bias_and_masked_parameter_exclusion(self):
        args = SimpleNamespace(max_num_categories=2, max_num_regressors=0, normalize='range')
        model_class = SimpleNamespace(global_param_names=['v', 'a', 'pi'])
        truth = np.array([[[-1., 99., -1.]], [[1., 99., 1.]]])
        pred = np.repeat((truth + 1)[:, None], 2, axis=1)
        bf = sweep.bayesflow_metrics()
        with patch.object(bf, 'root_mean_squared_error', wraps=bf.root_mean_squared_error) as compute:
            rows = sweep.config_metrics(args, model_class, {'1': ['v', 'pi']}, [],
                                        truth, pred, np.array([[1., 0., 1.]]))
        self.assertEqual(len(rows), 2)
        self.assertEqual({r['scope'] for r in rows}, {'local', 'pi'})
        for row in rows:
            self.assertEqual(row['NRMSE'], .5)
            self.assertEqual(row['Posterior Contraction'], 1.)
            self.assertAlmostEqual(row['Calibration Error'], .5)
            self.assertNotIn('Log Gamma', row)
        self.assertTrue(all(c.kwargs['normalize'] == 'range' for c in compute.call_args_list))

    def test_bayesflow_rmse_uses_posterior_rms_then_dataset_median(self):
        # Unequal posterior errors distinguish BF's actual reduction axes from
        # the former local implementation (which would give zero here).
        args = SimpleNamespace(max_num_categories=2, max_num_regressors=0, normalize='range')
        model_class = SimpleNamespace(global_param_names=['v'])
        truth = np.array([[[-1.]], [[1.]]])
        pred = np.array([[[[-1.]], [[-1.]], [[-1.]]], [[[1.]], [[1.]], [[4.]]]])
        row, = sweep.config_metrics(args, model_class, {'1': ['v']}, [],
                                    truth, pred, np.ones((1, 1)))
        self.assertAlmostEqual(row['NRMSE'], np.sqrt(3) / 4)
        self.assertAlmostEqual(row['Posterior Contraction'], .5)
        self.assertAlmostEqual(row['Calibration Error'], .5)

    def test_bayesflow_functions_match_direct_calls_on_supported_regression_cells(self):
        args = SimpleNamespace(max_num_categories=2, max_num_regressors=1, normalize='range')
        model_class = SimpleNamespace(global_param_names=['v', 'pi'])
        rng = np.random.default_rng(71)
        truth = rng.normal(size=(9, 2, 2))
        pred = truth[:, None] + rng.normal(size=(9, 11, 2, 2))
        mask = np.ones_like(truth)
        mask[:, 1, 1] = 0
        mask[:3, 1, 0] = 0
        truth[mask == 0] = 1e8
        for i in range(len(mask)):
            pred[i, :, mask[i] == 0] = -1e8
        rows = sweep.config_metrics(args, model_class, {'1': ['v', 'pi'], 'u_1': ['v']}, [],
                                    truth, pred, mask)
        self.assertEqual({r['parameter'] for r in rows}, {'v', 'pi', 'v -- $u_1$'})
        bf = sweep.bayesflow_metrics()
        for row in rows:
            r, c = {'v': (0, 0), 'pi': (0, 1), 'v -- $u_1$': (1, 0)}[row['parameter']]
            supported = mask[:, r, c] == 1
            estimates = pred[supported, :, r, c][..., None]
            targets = truth[supported, r, c][:, None]
            self.assertEqual(row['n_datasets'], int(supported.sum()))
            for fun in (bf.root_mean_squared_error, bf.posterior_contraction, bf.calibration_error):
                kwargs = {'normalize': 'range'} if fun == bf.root_mean_squared_error else {}
                result = fun(estimates=estimates, targets=targets, **kwargs)
                self.assertEqual(row[result['metric_name']], float(result['values'][0]))

    def test_report_equal_config_weight_and_separate_global(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'configs').mkdir()
            for index, values in enumerate(([0.], [1., 1., 1.])):
                records = [dict(parameter=f'v{j}', scope='local',
                                **dict(zip(sweep.METRICS, [v]*3))) for j, v in enumerate(values)]
                records.append(dict(parameter='pi', scope='pi', **dict(zip(sweep.METRICS, [.25]*3))))
                sweep.atomic_json(sweep.result_path(out, index),
                                  dict(index=index, family='DDM', metrics=records))
            summary = sweep.report(out, 2)
            local = next(r for r in summary if r['family'] == 'ALL' and r['scope'] == 'local')
            self.assertEqual(local['NRMSE'], .5)  # Not .75 from weighting by cell count.
            self.assertEqual(local['n_configs'], 2)
            global_row = next(r for r in summary if r['family'] == 'ALL' and r['scope'] == 'pi')
            self.assertEqual(global_row['NRMSE'], .25)
            latest = json.loads((out / 'latest_report.json').read_text())
            self.assertEqual(latest['n_configs'], 2)
            self.assertTrue(Path(latest['report']).exists())

    def test_realized_masks_exclude_unsupported_targets_and_draws(self):
        args = SimpleNamespace(max_num_categories=2, max_num_regressors=0, normalize='range')
        model_class = SimpleNamespace(global_param_names=['v', 'pi'])
        truth = np.array([[[999., -1.]], [[-1., 0.]], [[1., 1.]]])
        pred = np.repeat((truth + 1)[:, None], 4, axis=1)
        mask = np.ones_like(truth)
        mask[0, 0, 0] = 0
        rows = sweep.config_metrics(args, model_class, {'1': ['v', 'pi']}, [],
                                    truth, pred, mask)
        local, = [r for r in rows if r['scope'] == 'local']
        self.assertEqual(local['n_datasets'], 2)
        self.assertEqual(local['NRMSE'], .5)
        self.assertEqual(local['Posterior Contraction'], 1.)
        self.assertAlmostEqual(local['Calibration Error'], .5)
        global_row, = [r for r in rows if r['scope'] == 'pi']
        self.assertEqual(global_row['n_datasets'], 3)
        # Unsupported entries can be arbitrary without changing any metric.
        truth[0, 0, 0] = -123456.
        pred[0, :, 0, 0] = 777777.
        self.assertEqual(rows, sweep.config_metrics(args, model_class, {'1': ['v', 'pi']}, [],
                                                  truth, pred, mask))

    def test_cells_without_enough_support_are_omitted(self):
        args = SimpleNamespace(max_num_categories=2, max_num_regressors=0, normalize='range')
        model_class = SimpleNamespace(global_param_names=['v', 'pi'])
        truth = np.array([[[-1., -1.]], [[1., 1.]]])
        pred = np.repeat(truth[:, None], 4, axis=1)
        mask = np.ones_like(truth)
        mask[0, 0, 0] = 0
        rows = sweep.config_metrics(args, model_class, {'1': ['v', 'pi']}, [], truth, pred, mask)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['scope'], 'pi')

    def test_resume_rejects_changed_budget_before_starting_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            checkpoint = out / 'weights.pt'
            checkpoint.write_bytes(b'test checkpoint')
            args = SimpleNamespace(outdir=out, checkpoint=checkpoint, worker_index=None,
                                   gpu_ids='0,1', num_samples=200)
            old = sweep.settings(args)
            old['num_samples'] = 100
            sweep.atomic_json(out / 'run.json', dict(settings=old))
            with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': ''}), \
                    patch.object(sweep.subprocess, 'Popen') as spawn:
                with self.assertRaisesRegex(ValueError, 'different settings'):
                    sweep.coordinate(args)
                spawn.assert_not_called()

    def test_replay_preserves_exact_plan_and_rejects_changed_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            checkpoint = out / 'weights.pt'
            checkpoint.write_bytes(b'test checkpoint')
            args = SimpleNamespace(checkpoint=checkpoint, reference_run=out / 'run.json',
                                   n_configs=2, num_samples=200, report_every=200)
            reference_settings = sweep.settings(args)
            reference_settings['schema_version'] = 1
            reference_settings.pop('metric_implementation')
            reference_settings.pop('reference_run')
            reference_settings['n_configs'] = 3
            plan = [dict(index=i, family='DDM', design_config={'1': ['v']}) for i in range(3)]
            sweep.atomic_json(args.reference_run, dict(settings=reference_settings, plan=plan))
            self.assertEqual(sweep.build_plan(args), plan[:2])
            args.num_samples = 100
            with self.assertRaisesRegex(ValueError, 'different inference settings'):
                sweep.build_plan(args)

    def test_report_rejects_local_metric_records_in_a_bayesflow_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'configs').mkdir()
            sweep.atomic_json(out / 'run.json', dict(settings=dict(
                batch_size=200, metric_implementation=dict(name='bayesflow', version='2.0.14'))))
            sweep.atomic_json(sweep.result_path(out, 0), dict(index=0, family='DDM', metrics=[]))
            with self.assertRaisesRegex(ValueError, 'different metric implementation'):
                sweep.report(out, 1)

    def test_report_labels_regenerated_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'configs').mkdir()
            implementation = dict(name='bayesflow', version='2.0.14')
            sweep.atomic_json(out / 'run.json', dict(settings=dict(
                batch_size=200, normalize='range', metric_implementation=implementation,
                reference_run='old/run.json', reference_inputs='regenerate')))
            records = [dict(parameter='v', scope='local', **dict.fromkeys(sweep.METRICS, .5))]
            sweep.atomic_json(sweep.result_path(out, 0), dict(
                index=0, family='DDM', metrics=records, metric_implementation=implementation))
            sweep.report(out, 1)
            text = (out / 'reports/metrics_0001.md').read_text()
            self.assertIn('Metrics computed by BayesFlow 2.0.14', text)
            self.assertIn('Freshly simulated datasets', text)
            report = json.loads((out / 'reports/metrics_0001.json').read_text())
            self.assertEqual(report['reference_inputs'], 'regenerate')


if __name__ == '__main__':
    unittest.main()
