"""Validate the structural support diagnosis against real generator outputs."""
import copy
import unittest

import numpy as np

from cogformer.simulator.design_utils import build_random_design_config
from experiments.configs import MODELS
from experiments.diagnostics.design_support_audit import interaction_violations


class DesignSupportTests(unittest.TestCase):
    def test_benchmark_support_matrix(self):
        for family in ('ddm','cdm','rdm'):
            for case in ('intercept_only','fixed','regressed','fixed_regressed','interaction'):
                with self.subTest(family=family,case=case):
                    violations=interaction_violations(MODELS[family]['benchmark_configs'][case])
                    self.assertEqual(bool(violations),case in ('regressed','fixed_regressed','interaction'))

    def test_interaction_benchmark_excludes_forced_tau(self):
        for family in ('ddm','cdm','rdm'):
            violations=interaction_violations(MODELS[family]['benchmark_configs']['interaction'])
            self.assertEqual(violations,[dict(term='u_1:u_2',parameter='tau',kind='missing_forced_interaction')])

    def test_adding_tau_interaction_removes_this_violation(self):
        config=copy.deepcopy(MODELS['ddm']['benchmark_configs']['interaction'])
        config['u_1:u_2'].append('tau')
        self.assertEqual(interaction_violations(config),[])

    def test_empty_terms_and_unsupported_extra_interactions(self):
        config={'1':['tau'], 'u_1':['tau'], 'u_2':[], 'u_1:u_2':[]}
        self.assertEqual(interaction_violations(config),[])
        config['u_1:u_2']=['tau']
        self.assertEqual(interaction_violations(config)[0]['kind'],
                         'interaction_without_both_main_effects')

    def test_generated_configs_satisfy_rule(self):
        np.random.seed(20261025)
        for n in range(5):
            for _ in range(100):
                config=build_random_design_config(
                    intrinsic_params=['v','a','z','tau','s_v','s_tau','pi'],num_regressors=n,
                    free_intrinsics=['v','a','z','tau'],fixed_intrinsics=['s_v','s_tau'],
                    keep_intercept=True,add_interaction=True,global_params=['pi'])
                self.assertEqual(interaction_violations(config),[])


if __name__=='__main__':
    unittest.main()
