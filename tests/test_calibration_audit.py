"""Checks for bias direction, finite-sample coverage, and masked physical links."""
import unittest

import numpy as np

from experiments.diagnostics.calibration_audit import (
    diagnose, null_coverage_bounds, physical_times,
)


class CalibrationAuditTests(unittest.TestCase):
    def test_exchangeable_draws(self):
        rng = np.random.default_rng(42)
        draws = rng.normal(size=(10000, 200))
        truth = rng.normal(size=10000)
        stats, ranks, _ = diagnose(draws, truth, rng)
        self.assertLess(abs(ranks.mean()-.5), .015)
        lo, hi = null_coverage_bounds(np.array([.9]), 200)
        self.assertGreater(stats['coverage90'], lo[0]-.015)
        self.assertLess(stats['coverage90'], hi[0]+.015)

    def test_upward_bias_puts_truth_below_interval(self):
        rng = np.random.default_rng(7)
        draws = rng.normal(size=(1000, 200))+3
        truth = rng.normal(size=1000)
        stats, ranks, _ = diagnose(draws, truth, rng)
        self.assertGreater(stats['mean_bias'], 2.5)
        self.assertGreater(stats['truth_below90'], .7)
        self.assertLess(ranks.mean(), .1)

    def test_physical_links_mask_inactive_coefficients(self):
        coefficients = np.array([[[[0., 0.], [100., -100.]]]])
        mask = np.array([[True, True], [False, False]])
        reference = np.array([[1., 2.]])
        tau, stau = physical_times(coefficients, mask, reference, 0, 1)
        np.testing.assert_allclose(tau, np.log(2))
        np.testing.assert_allclose(stau, np.log(2))

    def test_link_after_linear_predictor(self):
        coefficients = np.array([[[[1., 0.], [2., 1.]]]])
        reference = np.array([[1., 3.]])
        tau, stau = physical_times(coefficients, np.ones((2,2),bool), reference, 0, 1)
        np.testing.assert_allclose(tau, np.logaddexp(0, 7.))
        np.testing.assert_allclose(stau, np.logaddexp(0, 3.))


if __name__ == '__main__':
    unittest.main()
