"""Delta heatmaps keep the six intrinsic columns and draw pi on its own."""

import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cogformer.diagnostics.plot.adaptive_correlation import (
    adaptive_correlation_delta, adaptive_metric_matrix,
)
from experiments.configs import MODELS, GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES
from experiments.utils import add_global_params


class DeltaLayoutTests(unittest.TestCase):
    cfg = add_global_params(MODELS["lba"], GLOBAL_PARAMS, GLOBAL_VARIABLE_NAMES)

    def mask(self, case):
        return np.array([[p in active for p in self.cfg["intrinsic_params"]]
                         for active in self.cfg["benchmark_configs"][case].values()], dtype=float)

    def test_delta_r_has_six_columns_in_every_case_and_pi_separately(self):
        params, names = self.cfg["intrinsic_params"], self.cfg["variable_names"]
        for case in ("fixed", "interaction"):
            with self.subTest(case=case):
                mask = self.mask(case)
                r = np.where(mask == 1, 0.5, np.nan)
                dc = self.cfg["benchmark_configs"][case]
                fig = adaptive_correlation_delta(r, r + 0.1, mask, dc, params, 2, variable_names=names,
                                                 global_params=GLOBAL_PARAMS, keep_all_cols=True)
                self.assertEqual(len(fig.axes[0].get_xticks()), 6)
                self.assertNotIn(r"$\pi$", [t.get_text() for t in fig.axes[0].get_xticklabels()])
                plt.close(fig)
                gfig = adaptive_correlation_delta(r, r + 0.1, mask, dc, params, 2, variable_names=names,
                                                  global_params=GLOBAL_PARAMS, global_only=True)
                self.assertEqual([t.get_text() for t in gfig.axes[0].get_xticklabels()], [r"$\pi$"])
                plt.close(gfig)

    def test_metric_matrix_keeps_inactive_columns_as_na(self):
        mask = self.mask("fixed")
        names = self.cfg["variable_names"]
        values = {names[c]: 0.1 for c in range(6) if mask[0, c]}
        matrix, display, _, _, cols, unplaced = adaptive_metric_matrix(
            values, mask, self.cfg["benchmark_configs"]["fixed"], self.cfg["intrinsic_params"], 2,
            variable_names=names, global_params=GLOBAL_PARAMS, keep_all_cols=True)
        self.assertEqual(cols, names[:6])
        self.assertEqual(unplaced, [])
        self.assertEqual(display[0].tolist(), mask[0, :6].tolist())


if __name__ == "__main__":
    unittest.main()
