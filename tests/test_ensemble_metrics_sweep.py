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
        with patch.object(sweep, 'adaptive_metrics', wraps=sweep.adaptive_metrics) as compute:
            rows = sweep.config_metrics(args, model_class, {'1': ['v', 'pi']}, [],
                                        truth, pred, np.array([[1., 0., 1.]]))
        self.assertEqual(len(rows), 2)
        self.assertEqual({r['scope'] for r in rows}, {'local', 'pi'})
        for row in rows:
            self.assertEqual(row['NRMSE'], .5)
            self.assertEqual(row['Posterior Contraction'], 1.)
            self.assertAlmostEqual(row['Calibration Error'], .5)
            self.assertNotIn('Log Gamma', row)
        self.assertTrue(all(c.kwargs['skip_log_gamma'] for c in compute.call_args_list))

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


if __name__ == '__main__':
    unittest.main()
