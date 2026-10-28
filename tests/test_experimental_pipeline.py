"""Exercise mitigation preprocessing and conditional-prior round trips end to end."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cogformer.simulator import ModelFamily,ModelClass
from experiments.configs import MODELS
from experiments.utils import add_global_params
from experiments.diagnostics.experimental_pipeline import ExperimentalPipeline,PipelineConfig
from experiments.diagnostics.mitigation_experiment import make_model
from experiments.utils import chunked_sample


def model_class():
    configs={name:add_global_params(MODELS[name],['pi'],[r'$\pi$']) for name in ('ddm','rdm','cdm')}
    families={name.upper():ModelFamily(model=cfg['model_cls'](),name=name.upper(),prior=cfg['prior_fn'](),
                                     intrinsic_params=cfg['intrinsic_params'],num_workers=1)
              for name,cfg in configs.items()}
    return ModelClass(families,{name.upper():cfg['link_fn']() for name,cfg in configs.items()}),configs


def batch_without_simulation(mc,configs):
    items=[]
    for family,cfg in configs.items():
        item=mc.model_families[family.upper()].sample(
            design_config=cfg['benchmark_configs']['full'],num_obs=24,num_regressors=2,
            max_num_regressors=2,max_num_categories=3,min_num_categories=3,
            categorical_regressors=['u_1','u_2'],keep_intercept=True,flatten_param_outputs=False,
            fixed_config=True,link_fun=cfg['link_fn'](),design_generator='gaussian_softmax',defer_simulation=True,
            mask_randomizer_kwargs={'fixed_values':cfg['default_fixed_values']})
        item.pop('_sim_request');item['obs']={'rts':np.ones((24,1),np.float32),'choices':np.zeros((24,1),np.float32)}
        item.update(num_obs=24,num_regressors=2,max_num_regressors=2,max_num_categories=3)
        items.append(item)
    return mc.collate(items)


class ExperimentalPipelineTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.mc,self.configs=model_class()
        self.samples=batch_without_simulation(self.mc,self.configs)

    def test_transform_decode_and_padding(self):
        pipeline=ExperimentalPipeline(self.mc,PipelineConfig(prior_coordinates=True))
        raw=self.samples['param_matrices'].reshape(3,-1,self.mc.n_global_params)
        z=pipeline.transport.apply(self.samples,raw)
        restored=pipeline.transport.apply(self.samples,z,inverse=True)
        np.testing.assert_allclose(restored,raw,atol=1e-12)
        mask=self.samples['param_masks'].reshape(raw.shape).astype(bool)
        self.assertTrue((z[~mask]==0).all())
        repeated=np.repeat(z[:,None],5,axis=1)
        decoded=pipeline.decode(self.samples,repeated)
        np.testing.assert_allclose(decoded,np.repeat(raw[:,None],5,axis=1),atol=1e-12)
        adapted=pipeline.adapt(self.samples)
        np.testing.assert_allclose(adapted['param_matrices'].numpy().reshape(z.shape),z,rtol=1e-6,atol=1e-6)

    def test_mask_count_and_observation_channels(self):
        samples=copy.deepcopy(self.samples)
        masks=samples['param_masks'].reshape(3,-1,self.mc.n_global_params)
        masks[:,1]=0;samples['design_matrices'][:,:,1]=999
        pipe=ExperimentalPipeline(self.mc,PipelineConfig(count_context=True,mask_inactive_columns=True))
        adapted=pipe.adapt(samples);x=adapted['input_data'].numpy()
        self.assertEqual(x.shape[-1],samples['design_matrices'].shape[-1]+6)
        np.testing.assert_array_equal(x[:,:,1],0)
        np.testing.assert_allclose(x[:,:,-1],np.log(24))
        np.testing.assert_array_equal(x[:,:,samples['design_matrices'].shape[-1]],1)

    def test_schema_rejects_incompatible_checkpoint(self):
        raw=ExperimentalPipeline(self.mc,PipelineConfig())
        normalized=ExperimentalPipeline(self.mc,PipelineConfig(prior_coordinates=True))
        raw.validate_schema(raw.schema())
        with self.assertRaises(ValueError):normalized.validate_schema(raw.schema())

    def test_simulation_truth_cannot_change_posterior_sampling(self):
        torch.set_num_threads(1)
        changed=copy.deepcopy(self.samples)
        values=changed['param_matrices'].reshape(3,-1,self.mc.n_global_params)
        values.fill(0)
        values[:,0,self.mc.global_param_names.index('pi')]=.1
        pipeline=ExperimentalPipeline(self.mc,PipelineConfig(True,True,True))
        original=pipeline.adapt(self.samples);alternative=pipeline.adapt(changed)
        self.assertFalse(torch.equal(original['param_matrices'],alternative['param_matrices']))
        model=make_model(with_count=True).eval();model._compiled_vel=model.decoder.velocity
        draws=[]
        with patch('cogformer.networks.transformers.cogformer.tqdm',lambda iterator,**kw:iterator):
            for adapted in (original,alternative):
                torch.manual_seed(543)
                draws.append(chunked_sample(model,adapted,num_samples=2,steps=2,solver='rk4',compact=False))
        np.testing.assert_array_equal(draws[0],draws[1])
        np.testing.assert_array_equal(pipeline.decode(self.samples,draws[0]),pipeline.decode(changed,draws[1]))

    def test_label_free_inference_preserves_all_conditioning(self):
        label_free=copy.deepcopy(self.samples);label_free.pop('param_matrices')
        for config in (PipelineConfig(),PipelineConfig(True,True,True)):
            pipeline=ExperimentalPipeline(self.mc,config)
            training=pipeline.adapt(self.samples)
            inference=pipeline.adapt(label_free,inference=True)
            self.assertNotIn('param_matrices',inference)
            self.assertEqual(set(training)-{'param_matrices'},set(inference))
            for key,value in inference.items():
                if isinstance(value,torch.Tensor):torch.testing.assert_close(value,training[key],rtol=0,atol=0)
                else:self.assertEqual(value,training[key])


if __name__=='__main__':unittest.main()
