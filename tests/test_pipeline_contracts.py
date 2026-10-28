"""Trace parameter links and model-local/global collation without diffusion simulation."""
import unittest
import numpy as np

from cogformer.adapter import Adapter
from cogformer.simulator import ModelFamily,ModelClass
from experiments.configs import MODELS
from experiments.utils import add_global_params


class PipelineContractTests(unittest.TestCase):
    def test_parameter_links_and_collation(self):
        configs={name:add_global_params(MODELS[name],['pi'],[r'$\pi$']) for name in ('ddm','rdm','cdm')}
        families={name.upper():ModelFamily(model=cfg['model_cls'](),name=name.upper(),
                                           prior=cfg['prior_fn'](),intrinsic_params=cfg['intrinsic_params'],num_workers=1)
                  for name,cfg in configs.items()}
        mc=ModelClass(families,{name.upper():cfg['link_fn']() for name,cfg in configs.items()})
        for generator in ('mixture','gaussian_softmax'):
            for family,cfg in configs.items():
                for case in ('fixed','interaction'):
                    with self.subTest(generator=generator,family=family,case=case):
                        np.random.seed(42)
                        names=cfg['intrinsic_params'];links=cfg['link_fn']()
                        item=families[family.upper()].sample(
                            design_config=cfg['benchmark_configs'][case],num_obs=24,num_regressors=2,
                            max_num_regressors=2,max_num_categories=3,min_num_categories=3,
                            categorical_regressors=['u_1'],keep_intercept=True,add_interaction=True,
                            fixed_config=True,flatten_param_outputs=False,link_fun=links,
                            design_generator=generator,defer_simulation=True,
                            mask_randomizer_kwargs={'fixed_values':cfg['default_fixed_values']})
                        params,context=item.pop('_sim_request')
                        x=item['design_matrix'];coeff=item['param_matrix'];mask=item['param_mask']
                        predictor=x@coeff
                        expected={}
                        for j,name in enumerate(names):
                            if name in cfg['default_fixed_values'] and not mask[:,j].any():
                                expected[name]=np.full(24,cfg['default_fixed_values'][name])
                            else:expected[name]=links[name](predictor[:,j])
                        expected={key:np.asarray(value,dtype=np.float32) for key,value in expected.items()}
                        if family=='rdm':
                            drift=np.repeat(expected['v'][:,None],2,axis=1)
                            favored=np.asarray(context.get('favored_idx',np.ones(24,dtype=int))).astype(int)
                            drift[np.arange(24),favored]+=expected.pop('v_diff')
                            expected['v']=drift
                        elif family=='cdm':
                            angle=expected.pop('v_theta');radius=expected['v']
                            expected['v']=np.stack([radius*np.cos(angle),radius*np.sin(angle)],axis=1)
                        for name in params:
                            np.testing.assert_allclose(params[name],expected[name],rtol=2e-6,atol=1e-7)
                        # An echo observation isolates adapter/collation from solver noise.
                        item['obs']={'rts':(params['tau']+.1)[:,None],
                                     'choices':np.zeros((24,1),dtype=np.float32)}
                        item.update(num_obs=24,num_regressors=2,max_num_regressors=2,max_num_categories=3)
                        batch=mc.collate([item]);adapted=Adapter().adapt(batch,device='cpu')
                        nrows=coeff.shape[0]
                        global_coeff=batch['param_matrices'].reshape(1,nrows,mc.n_global_params)
                        global_mask=batch['param_masks'].reshape(1,nrows,mc.n_global_params)
                        np.testing.assert_array_equal(global_coeff[0][:,mc.local_to_global[family.upper()]],coeff.astype(np.float32))
                        np.testing.assert_array_equal(global_mask[0][:,mc.local_to_global[family.upper()]],mask)
                        encoded=adapted['input_data'].numpy()
                        np.testing.assert_allclose(encoded[0,:,:nrows],x,rtol=1e-6,atol=1e-6)
                        np.testing.assert_array_equal(encoded[0,:,nrows],item['obs']['rts'][:,0])
                        if 'favored_idx' in item['per_trial_context']:
                            np.testing.assert_allclose(encoded[0,:,nrows+3],item['per_trial_context']['favored_idx']/3)


if __name__=='__main__':unittest.main()
