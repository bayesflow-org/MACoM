"""Statistical and experimental controls for mitigation comparisons."""
import unittest
import numpy as np
import pandas as pd
import torch

from experiments.diagnostics.analyze_mitigation_experiment import paired_bootstrap,all_parameter_scores,METRICS
from experiments.diagnostics.mitigation_rank_metrics import randomized_central_coverage,randomized_rank_histogram
from experiments.diagnostics.mitigation_experiment import make_model,matched_initialization,build_class,generate_batch
from experiments.diagnostics.experimental_pipeline import ExperimentalPipeline,PipelineConfig
from experiments.diagnostics.mitigation_artifacts import verify_training_banks
from experiments.diagnostics.verify_mitigation_artifacts import payload_hash


class MitigationAnalysisTests(unittest.TestCase):
    def test_payload_hash_covers_stimulus_context_and_types(self):
        first={'context':{'favored_idx':np.array([[0,1]],dtype=np.int64)},'theta':np.ones((1,2))}
        copied={'theta':np.ones((1,2)), 'context':{'favored_idx':np.array([[0,1]],dtype=np.int64)}}
        self.assertEqual(payload_hash(first),payload_hash(copied))
        copied['context']['favored_idx'][0,0]=1
        self.assertNotEqual(payload_hash(first),payload_hash(copied))
        copied['context']['favored_idx']=np.array([[0,1]],dtype=np.float64)
        self.assertNotEqual(payload_hash(first),payload_hash(copied))

    def test_repeated_training_seed_requires_same_bank(self):
        first=dict(seeds=[33],training_data_hashes={'33':{'forced':'a','optional':'b'}})
        second=dict(seeds=[33],training_data_hashes={'33':{'forced':'a','optional':'changed'}})
        self.assertEqual(verify_training_banks([first,first])[33],first['training_data_hashes']['33'])
        with self.assertRaises(ValueError):verify_training_banks([first,second])
        with self.assertRaises(ValueError):verify_training_banks([dict(seeds=[33],training_data_hashes={})])

    def test_finite_draw_rank_coverage_has_exact_null(self):
        for d in (8,64,128,200):
            draws=np.tile(np.arange(d),(d+1,1))
            truth=np.arange(d+1)-.5
            value=randomized_central_coverage(draws,truth)
            self.assertAlmostEqual(value.mean(),.9,places=14)
        # Exact ties also span a uniform randomized rank interval.
        self.assertAlmostEqual(randomized_central_coverage(np.zeros((1,128)),np.array([0.]))[0],.9)

    def test_randomized_histogram_has_exact_uniform_null(self):
        for d in (8,64,128):
            draws=np.tile(np.arange(d),(d+1,1));truth=np.arange(d+1)-.5
            histogram=randomized_rank_histogram(draws,truth)
            np.testing.assert_allclose(histogram.mean(0),np.full(10,.1),atol=1e-14)
            np.testing.assert_allclose(histogram.sum(1),1,atol=1e-14)
        np.testing.assert_allclose(randomized_rank_histogram(np.zeros((1,128)),np.array([0.])),.1)

    def test_bootstrap_preserves_constant_paired_effect(self):
        records=[dict(seed=seed,family=family,trials=n,dataset=dataset,delta=-.1)
                 for seed in [11,22,33] for family in ['ddm','rdm'] for n in [200,600]
                 for dataset in range(5)]
        low,high=paired_bootstrap(pd.DataFrame(records),[11,22,33],np.random.default_rng(4),100)
        np.testing.assert_allclose([low,high],[-.1,-.1],atol=1e-14)

    def test_overall_score_weights_cells_and_excludes_functionals(self):
        records=[]
        for param,role,cells,value in [('pi','intercept',1,.2),('v','main_effect',2,.5),('ndt_mean','origin',1,100.)]:
            row=dict(seed=11,arm='reference',population='forced',family='ddm',trials=200,dataset=0,
                     parameter=param,role=role,cells=cells)
            row.update({metric:value for metric in METRICS});records.append(row)
        overall=all_parameter_scores(pd.DataFrame(records)).iloc[0]
        self.assertAlmostEqual(overall.normalized_crps,.4)
        self.assertEqual(overall.cells,3)
        self.assertEqual(overall.invalid_pi_mass,.2)

    def test_count_arm_starts_from_matched_function(self):
        torch.set_num_threads(1);torch.manual_seed(23)
        reference=make_model();count=make_model(with_count=True)
        matched_initialization(count,reference.state_dict())
        reference.eval();count.eval()
        x=torch.randn(2,30,14);extra=torch.full((2,30,1),np.log(30))
        with torch.inference_mode():
            a=reference.encoder(x);b=count.encoder(torch.cat([x,extra],dim=-1))
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)

    def test_paired_simulations_preserve_retained_coefficients(self):
        mc,configs=build_class()
        base,optional=generate_batch(mc,configs,6,32,1234,('ddm','rdm','cdm'),balanced=True)
        np.testing.assert_array_equal(base['design_matrices'],optional['design_matrices'])
        keep=optional['param_masks'].astype(bool)
        self.assertTrue((optional['param_masks']<=base['param_masks']).all())
        np.testing.assert_array_equal(optional['param_matrices'][keep],base['param_matrices'][keep])
        for config in [PipelineConfig(),PipelineConfig(True,True,True)]:
            pipeline=ExperimentalPipeline(mc,config)
            adapted=pipeline.adapt(optional)
            self.assertTrue(torch.isfinite(adapted['param_matrices']).all())


if __name__=='__main__':unittest.main()
