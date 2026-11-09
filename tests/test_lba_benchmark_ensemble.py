"""CogFormer LBA bench5 inputs must be the compact grid the BF layout expects."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.configs import MODELS, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES
from experiments.experiment_1a.model_family_bf_benchmark_ensemble import CASES
from experiments.experiment_3.lba_benchmark_ensemble import ARMS, load_case
from experiments.utils import add_global_params


class LbaBenchmarkEnsembleTests(unittest.TestCase):
    cfg = add_global_params(MODELS["lba"], GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)

    def grid(self, case):
        return np.array([[p in active for p in self.cfg["intrinsic_params"]]
                         for active in self.cfg["benchmark_configs"][case].values()])

    def write(self, directory, case, infix, mask=None, cats=2):
        grid = self.grid(case)
        truth = np.where(grid, np.arange(grid.size).reshape(grid.shape) + 1., 0.)
        truth = np.broadcast_to(truth, (3, *grid.shape)).copy()
        pred = np.repeat(truth[:, None], 4, axis=1)
        np.savez(directory / f"lba_{case}_{infix}.npz", true_set=truth, pred_set=pred,
                 params_mask=grid if mask is None else mask, max_num_categories=cats)
        return grid, truth

    def test_every_arm_and_case_loads_on_the_design_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for infix in ARMS.values():
                for case in CASES:
                    with self.subTest(infix=infix, case=case):
                        grid, truth = self.write(directory, case, infix)
                        entry = load_case(directory, case, infix, self.cfg)
                        np.testing.assert_array_equal(entry["mask"], grid)
                        np.testing.assert_array_equal(entry["true"], truth)
                        self.assertEqual(entry["pred"].shape, (3, 4, *grid.shape))
                        # pi is active only at the intercept.
                        self.assertEqual(entry["mask"][:, -1].tolist(), [True, False, False, False])

    def test_rejects_wrong_mask_or_padded_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            wrong = self.grid("fixed").copy()
            wrong[0, [0, 4]] = wrong[0, [4, 0]]
            self.write(directory, "fixed", "ft_pred", mask=wrong)
            with self.assertRaisesRegex(ValueError, "compact bench5 grid"):
                load_case(directory, "fixed", "ft_pred", self.cfg)
            self.write(directory, "fixed", "ft_pred", cats=4)
            with self.assertRaisesRegex(ValueError, "compact bench5 grid"):
                load_case(directory, "fixed", "ft_pred", self.cfg)


if __name__ == "__main__":
    unittest.main()
