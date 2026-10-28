"""Reported predictor types must match the matrix used by the coefficient prior."""
import unittest
import numpy as np
from experiments.diagnostics.mitigation_experiment import build_class,seed_simulator
from experiments.diagnostics.experimental_pipeline import ExperimentalPipeline,PipelineConfig


class DiscreteMetadataTests(unittest.TestCase):
    def test_random_predictor_metadata_matches_realized_design(self):
        mc,configs=build_class();family=mc.model_families['DDM'];cfg=configs['ddm']
        for generator in ['mixture','gaussian_softmax']:
            for seed in range(20):
                with self.subTest(generator=generator,seed=seed):
                    np.random.seed(seed);seed_simulator(seed+12345)
                    item=family.sample(num_obs=32,num_regressors=2,max_num_regressors=2,max_num_categories=3,
                        keep_intercept=True,flatten_param_outputs=False,add_interaction=True,link_fun=cfg['link_fn'](),
                        design_generator=generator,defer_simulation=True,mask_randomizer_kwargs=dict(
                            free_intrinsics=['v','a','z','tau'],fixed_intrinsics=['s_v','s_tau'],
                            fixed_values=cfg['default_fixed_values'],global_params=['pi']))
                    for reg in range(2):
                        first=1+reg*2
                        categorical=np.isin(item['design_matrix'][:,first],[0,1]).all()
                        self.assertEqual(bool(item['discrete_mask'][reg]),categorical)
                    item.pop('_sim_request');item['obs']={'rts':np.ones((32,1)),'choices':np.zeros((32,1))}
                    item.update(num_obs=32,num_regressors=2,max_num_regressors=2,max_num_categories=3)
                    batch=mc.collate([item])
                    pipeline=ExperimentalPipeline(mc,PipelineConfig(prior_coordinates=True))
                    raw=batch['param_matrices'].reshape(1,-1,mc.n_global_params)
                    transformed=pipeline.transport.apply(batch,raw)
                    np.testing.assert_allclose(pipeline.transport.apply(batch,transformed,inverse=True),raw,atol=1e-12)


if __name__=='__main__':unittest.main()
