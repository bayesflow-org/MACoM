"""Checks for matched-trajectory comparisons, independent of checkpoint files."""
import unittest
import numpy as np
from experiments.diagnostics.sampler_audit import trajectory_comparison


class TrajectoryTests(unittest.TestCase):
    def test_identical_trajectories_have_zero_error(self):
        draws=np.array([[0.,1.,2.],[2.,3.,4.]])
        stats=trajectory_comparison(draws,draws)
        self.assertTrue(all(x==0 for x in stats.values()))

    def test_direction_and_scale(self):
        reference=np.array([[-1.,0.,1.],[-1.,0.,1.]])
        stats=trajectory_comparison(reference+.2,reference)
        self.assertAlmostEqual(stats['trajectory_rms_sd'],.2)
        self.assertAlmostEqual(stats['mean_shift_sd'],.2)
        self.assertAlmostEqual(stats['trajectory_rms'],.2)

    def test_sampler_reuses_initial_noise_across_solvers(self):
        import torch
        from unittest.mock import patch
        from cogformer.networks.transformers.cogformer import CogFormer
        torch.set_num_threads(1)
        model=CogFormer(encoder_input_dim=3,encoder_embed_dim=8,seed_dim=8,
                        encoder_num_layers=1,decoder_num_layers=1,
                        encoder_num_heads=2,decoder_num_heads=2,num_seeds=2,
                        proj_dim=8,time_embedding_dim=4,pos_embedding_dim=4,
                        weight_init='default')
        model.eval()
        model._compiled_vel=model.decoder.velocity
        inputs=torch.zeros(1,5,3)
        params=torch.zeros(1,2,1,dtype=torch.long)
        regressors=torch.zeros(1,2,5,dtype=torch.long)
        masks=torch.ones(1,2)
        original=torch.randn
        noises=[]
        def capture(*args,**kwargs):
            result=original(*args,**kwargs)
            noises.append(result.clone())
            return result
        for solver in ('midpoint','rk4'):
            torch.manual_seed(123)
            with torch.inference_mode(), patch('torch.randn',side_effect=capture):
                model.sample(inputs,params,regressors,masks,
                             steps=2,num_samples=3,solver=solver)
        self.assertEqual(len(noises),2)
        torch.testing.assert_close(noises[0],noises[1],rtol=0,atol=0)

    def test_degenerate_reference_rejected(self):
        with self.assertRaises(ValueError):
            trajectory_comparison(np.ones((2,3)),np.ones((2,3)))


if __name__=='__main__':
    unittest.main()
