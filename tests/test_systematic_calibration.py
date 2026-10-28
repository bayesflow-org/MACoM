"""Statistical checks for the all-parameter audit."""
import unittest
import numpy as np
from experiments.diagnostics.systematic_calibration import bh_adjust,holm_adjust,ranks,radial_ranks,cv_correction,coverage
from experiments.diagnostics.calibration_audit import diagnose


class SystematicCalibrationTests(unittest.TestCase):
    def test_multiple_test_corrections(self):
        p=np.array([.01,.04,.03,.5])
        np.testing.assert_allclose(bh_adjust(p),[.04,.0533333333333,.0533333333333,.5])
        np.testing.assert_allclose(holm_adjust(p),[.04,.09,.09,.5])

    def test_radial_rank_uniformity_without_gaussian_assumption(self):
        rng=np.random.default_rng(25)
        # Heavy-tailed, independent exchangeable truths/posterior samples.
        sample=rng.standard_t(5,size=(3000,121,3))
        r=radial_ranks(sample[:,:120],sample[:,120])
        self.assertLess(abs(r.mean()-.5),.02)
        self.assertLess(abs((r>.95).mean()-.05),.015)

    def test_tie_randomization(self):
        rng=np.random.default_rng(2)
        draws=rng.integers(0,4,size=(5000,200));truth=rng.integers(0,4,size=5000)
        _,randomized,_=diagnose(draws,truth,rng)
        self.assertLess(abs(randomized.mean()-.5),.015)
        self.assertLess(abs((randomized<.1).mean()-.1),.015)
        self.assertLess(abs(ranks(draws,truth).mean()-.5),.015)

    def test_cross_validated_moment_diagnostic_detects_location_error(self):
        rng=np.random.default_rng(21)
        truth=rng.normal(size=3000);draws=rng.normal(size=(3000,200))+2
        shifted,_=cv_correction(draws,truth)
        # A fixed +2 error is removed using other datasets, not each row's truth.
        self.assertGreater(coverage(shifted,truth).mean(),.85)
        self.assertLess(coverage(draws,truth).mean(),.5)
        # Within-dataset draw differences must remain unchanged by location shift.
        np.testing.assert_allclose(shifted[:,1]-shifted[:,0],draws[:,1]-draws[:,0],atol=1e-14)


if __name__=='__main__':unittest.main()
