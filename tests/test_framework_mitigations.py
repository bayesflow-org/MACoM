"""Check invariance and preservation of generative priors in experimental proposals."""
import unittest
import numpy as np
from scipy.stats import norm

from experiments.diagnostics.prior_transport import (
    beta1_to_normal,normal_to_beta1,gaussian_to_normal,normal_to_gaussian,
    categorical_contrast_covariance,categorical_interaction_covariance,
)
from experiments.diagnostics.framework_proposals import optional_interactions,design_neutral_encoder_inputs


class FrameworkMitigationTests(unittest.TestCase):
    def test_beta_transport_roundtrip_and_support(self):
        z=np.linspace(-8,8,1000);p=normal_to_beta1(z)
        self.assertTrue(np.all((p>0)&(p<1)))
        np.testing.assert_allclose(beta1_to_normal(p),z,atol=1e-12)
        p=np.array([1e-12,.0001,.01,.1,.9,1-1e-8])
        np.testing.assert_allclose(normal_to_beta1(beta1_to_normal(p)),p,rtol=1e-12,atol=1e-15)

    def test_beta_prior_is_preserved(self):
        rng=np.random.default_rng(53)
        z=beta1_to_normal(rng.beta(1,50,100000))
        self.assertLess(abs(z.mean()),.015)
        self.assertLess(abs(z.std()-1),.015)
        for q in (.05,.5,.95):
            self.assertLess(abs(np.mean(z<norm.ppf(q))-q),.01)
        with self.assertRaises(ValueError):beta1_to_normal(np.array([-.01,.2]))

    def test_correlated_categorical_whitening(self):
        rng=np.random.default_rng(1)
        cells=rng.normal(size=(20000,3,4))
        contrasts=(cells[:,1:,1:]-cells[:,1:,:1]-cells[:,:1,1:]+cells[:,:1,:1]).reshape(20000,-1)
        covariance=categorical_interaction_covariance(3,4)
        np.testing.assert_allclose(np.cov(contrasts,rowvar=False),covariance,atol=.1)
        z=gaussian_to_normal(contrasts,np.zeros(6),covariance)
        np.testing.assert_allclose(np.cov(z,rowvar=False),np.eye(6),atol=.03)
        np.testing.assert_allclose(normal_to_gaussian(z,np.zeros(6),covariance),contrasts,atol=1e-12)
        self.assertEqual(categorical_contrast_covariance(3)[0,1],1)

    def test_optional_interactions_cover_all_eligible_subsets(self):
        rng=np.random.default_rng(2)
        config={'1':['v','a','tau'],'u_1':['v','a','tau'],'u_2':['v','a','tau'],
                'u_1:u_2':['v','a','tau']}
        observed={tuple(optional_interactions(config,rng)['u_1:u_2']) for _ in range(1000)}
        self.assertEqual(len(observed),8)
        self.assertIn((),observed)
        self.assertIn(('v','a','tau'),observed)
        self.assertEqual(config['u_1:u_2'],['v','a','tau'])

    def test_nuisance_invariance_and_count_information(self):
        x=np.arange(2*10*6,dtype=float).reshape(2,10,6)
        masks=np.array([[1,0,0,0,1,0],[1,0,0,0,1,0]]) # three design columns x two parameters
        a=design_neutral_encoder_inputs(x,masks,2,3)
        altered=x.copy();altered[:,:,1]=100000
        b=design_neutral_encoder_inputs(altered,masks,2,3)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a[:,:,3:-1],x[:,:,3:])
        doubled=design_neutral_encoder_inputs(np.repeat(x,2,axis=1),masks,2,3)
        self.assertAlmostEqual(doubled[0,0,-1]-a[0,0,-1],np.log(2))
        with self.assertRaises(ValueError):design_neutral_encoder_inputs(x,masks,3,3)


if __name__=='__main__':unittest.main()
