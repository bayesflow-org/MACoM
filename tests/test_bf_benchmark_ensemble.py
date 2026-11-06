"""Check the scientific mapping from flat BF outputs to shared coefficient cells."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.configs import MODELS, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES
from experiments.experiment_1a.model_family_bf_benchmark_ensemble import CASES, FAMILIES, load_case
from experiments.utils import add_global_params


class BayesFlowBenchmarkEnsembleTests(unittest.TestCase):
    def write_case(self, directory, family, case, cfg):
        mask = np.array([[p in active for p in cfg["intrinsic_params"]]
                         for active in cfg["benchmark_configs"][case].values()])
        # Each value identifies its row and column; inactive cells deliberately
        # contain garbage to catch accidental inclusion in diagnostics.
        values = np.arange(mask.size).reshape(mask.shape) + 1.
        truth = np.broadcast_to(values, (3, *values.shape)).copy()
        truth[:, ~mask] = np.nan
        if not mask[1:].any():
            truth = truth[:, :1]
        truth = truth.reshape(3, -1)
        pred = np.repeat(truth[:, None], 4, axis=1)
        masks = np.broadcast_to(mask.ravel()[:truth.shape[-1]], truth.shape).copy()
        path = directory / f"{family}_{case}_data.npz"
        np.savez(path, true_set=truth, pred_set=pred, param_masks=masks)
        return path, mask, values, truth, pred, masks

    def test_every_family_and_case_preserves_cell_identity_and_excludes_inactive(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for family in FAMILIES:
                cfg = add_global_params(MODELS[family], GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
                for case in CASES:
                    with self.subTest(family=family, case=case):
                        _, mask, values, *_ = self.write_case(directory, family, case, cfg)
                        entry = load_case(directory, family, case, cfg)
                        expected = np.where(mask, values, 0.)
                        np.testing.assert_array_equal(entry["mask"], mask)
                        np.testing.assert_array_equal(entry["true"], np.broadcast_to(expected, (3, *mask.shape)))
                        np.testing.assert_array_equal(entry["pred"], np.broadcast_to(expected, (3, 4, *mask.shape)))
                        # pi has its own column and is active only at the intercept.
                        self.assertEqual(entry["mask"][:, -1].tolist(), [True, False, False, False])

    def test_rejects_misaligned_or_heterogeneous_masks(self):
        cfg = add_global_params(MODELS["ddm"], GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path, _, _, truth, pred, masks = self.write_case(directory, "ddm", "fixed", cfg)
            # Swap active/inactive slots without changing the active count.
            wrong = masks.copy()
            wrong[:, [0, 4]] = wrong[:, [4, 0]]
            np.savez(path, true_set=truth, pred_set=pred, param_masks=wrong)
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_case(directory, "ddm", "fixed", cfg)
            masks[1, 0] = False
            np.savez(path, true_set=truth, pred_set=pred, param_masks=masks)
            with self.assertRaisesRegex(ValueError, "identical parameter mask"):
                load_case(directory, "ddm", "fixed", cfg)


if __name__ == "__main__":
    unittest.main()
