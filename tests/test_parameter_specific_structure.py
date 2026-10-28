"""Structural-prior and train/inference contracts for parameter-specific effects."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cogformer.adapter import Adapter
from cogformer.networks.transformers.cogformer import CogFormer
from cogformer.simulator import ModelFamily, ModelClass
from cogformer.simulator.design_utils import build_random_design_config
from experiments.configs import MODELS
from experiments.utils import (add_global_params, chunked_sample, ModelClassDataset, structure_training_schema,
                               validate_structure_resume)


def small_model(input_dim=8, conditioned=True):
    return CogFormer(encoder_input_dim=input_dim, encoder_embed_dim=16, encoder_num_layers=1,
                     decoder_num_layers=2, encoder_num_heads=2, decoder_num_heads=2,
                     num_seeds=4, seed_dim=16, proj_dim=32, time_embedding_dim=8,
                     pos_embedding_dim=8, num_models=1, gated_residual=False,
                     pre_pool_ffn=False, weight_init='default',
                     encoder_structure_conditioning=conditioned)


def family():
    cfg=add_global_params(MODELS['ddm'], ['pi'], [r'$\pi$'])
    mf=ModelFamily(model=cfg['model_cls'](), name='DDM', prior=cfg['prior_fn'](),
                   intrinsic_params=cfg['intrinsic_params'], num_workers=1)
    return mf,cfg


class RandomInteractionTests(unittest.TestCase):
    def config(self, **kwargs):
        options=dict(intrinsic_params=['v','tau','pi'],num_regressors=2,free_intrinsics=['v','tau'],
                     global_params=['pi'],keep_intercept=True,free_prob=1.,add_interaction=True)
        options.update(kwargs)
        return build_random_design_config(**options)

    def test_parameter_specific_subsets_and_probability_endpoints(self):
        np.random.seed(44)
        observed={tuple(self.config(interaction_policy='optional',interaction_prob=.5)['u_1:u_2'])
                  for _ in range(128)}
        self.assertEqual(observed,{(),('v',),('tau',),('v','tau')})
        for probability,expected in [(0,[]),(1,['v','tau'])]:
            config=self.config(interaction_policy='optional',interaction_prob=probability)
            self.assertEqual(config['u_1:u_2'],expected)
            self.assertEqual(config['u_1'],['v','tau'])
            self.assertEqual(config['u_2'],['v','tau'])

    def test_shared_beta_probability_retains_all_subset_support(self):
        np.random.seed(53)
        observed={tuple(self.config(interaction_policy='optional')['u_1:u_2']) for _ in range(256)}
        self.assertEqual(observed,{(),('v',),('tau',),('v','tau')})

    def test_pairwise_heredity_and_global_exclusion(self):
        np.random.seed(17)
        for _ in range(100):
            config=self.config(num_regressors=4,free_prob=None,interaction_policy='optional')
            for key,parameters in config.items():
                if ':' in key:
                    left,right=key.split(':')
                    self.assertLessEqual(set(parameters),set(config[left])&set(config[right]))
                if key!='1':self.assertNotIn('pi',parameters)

    def test_legacy_and_disabled_interactions_keep_rng_behavior(self):
        for enabled in (False,True):
            np.random.seed(92);default=self.config(add_interaction=enabled);next_default=np.random.rand()
            np.random.seed(92);explicit=self.config(add_interaction=enabled,interaction_policy='all');next_explicit=np.random.rand()
            self.assertEqual(default,explicit);self.assertEqual(next_default,next_explicit)
        np.random.seed(92);disabled=self.config(add_interaction=False)
        np.random.seed(92);optional=self.config(add_interaction=False,interaction_policy='optional')
        self.assertEqual(disabled,optional)
        for n in (0,1):self.config(num_regressors=n,interaction_policy='optional')

    def test_invalid_controls_raise(self):
        for options in [dict(interaction_policy='unknown'),dict(interaction_prob=.5),
                        *[dict(interaction_policy='optional',interaction_prob=p) for p in (-1,2,float('nan'))]]:
            with self.subTest(options=options),self.assertRaises(ValueError):self.config(**options)

    def test_explicit_parameter_structure_and_physical_parameters_preserved(self):
        mf,cfg=family()
        config={'1':['v','a','z','tau','pi'],'u_1':['v','tau'],'u_2':['v','tau'],'u_1:u_2':['v']}
        original=copy.deepcopy(config)
        item=mf.sample(design_config=config,num_obs=32,num_regressors=2,max_num_regressors=2,
                       max_num_categories=2,categorical_regressors=[],flatten_param_outputs=False,
                       link_fun=cfg['link_fn'](),defer_simulation=True,
                       interaction_policy='optional',interaction_prob=0.)
        self.assertEqual(config,original)
        interaction=item['position_codes'][:,1]>0
        jv=mf.intrinsic_params.index('v');jt=mf.intrinsic_params.index('tau')
        self.assertTrue(item['param_mask'][interaction,jv].all())
        self.assertFalse(item['param_mask'][interaction,jt].any())
        prepared,_=item['_sim_request']
        expected=cfg['link_fn']()['tau'](item['design_matrix']@item['param_matrix'][:,jt])
        np.testing.assert_allclose(prepared['tau'],expected,rtol=2e-6)


class StructureEncoderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(32)
        self.model=small_model().eval();self.x=torch.randn(1,12,8)
        self.params=torch.tensor([[[0],[1],[0],[1]]])
        self.regs=torch.tensor([[[1,0,1,0,1],[1,0,1,0,1],[2,0,1,0,2],[2,0,1,0,2]]])
        self.mask=torch.tensor([[1.,0.,0.,1.]])

    def test_joint_assignments_with_same_marginals_are_distinguished(self):
        first=self.model.encode_conditions(self.x,self.params,self.regs,self.mask)
        second=self.model.encode_conditions(self.x,self.params,self.regs,1-self.mask)
        self.assertGreater(float((first-second).detach().abs().max()),1e-5)

    def test_inactive_tokens_compaction_and_permutations(self):
        reference=self.model.encode_conditions(self.x,self.params,self.regs,self.mask)
        active=self.mask[0].bool()
        compact=self.model.encode_conditions(self.x,self.params[:,active],self.regs[:,active],self.mask[:,active])
        torch.testing.assert_close(compact,reference,rtol=1e-5,atol=1e-6)
        changed=self.regs.clone();changed[:,~active]=torch.tensor([3,4,2,3,20])
        altered=self.model.encode_conditions(self.x,self.params,changed,self.mask)
        torch.testing.assert_close(altered,reference,rtol=0,atol=0)
        order=torch.tensor([3,1,0,2])
        reordered=self.model.encode_conditions(self.x.flip(1),self.params[:,order],self.regs[:,order],self.mask[:,order])
        torch.testing.assert_close(reordered,reference,rtol=1e-5,atol=1e-6)

    def test_legacy_path_and_required_mask(self):
        model=small_model(conditioned=False).eval()
        self.assertFalse(any(key.startswith('structure_embedding.') for key in model.state_dict()))
        torch.testing.assert_close(model.encode_conditions(self.x,self.params,self.regs,self.mask),model.encoder(self.x),rtol=0,atol=0)
        with self.assertRaises(ValueError):self.model.encode_conditions(self.x,self.params,self.regs)

    def test_real_simulator_class_training_and_sampling(self):
        mf,cfg=family();mc=ModelClass({'DDM':mf},{'DDM':cfg['link_fn']()},num_workers=1)
        mf.mask_randomizer_kwargs=dict(free_intrinsics=cfg['free_params'],fixed_intrinsics=['s_v','s_tau'],
                                      fixed_values=cfg['default_fixed_values'],global_params=['pi'],intercept_prob=0.)
        with patch.dict(os.environ,{'COGFORMER_SIM_BACKEND':'numba'}):
            # Exercise both batch APIs, including native propagation of the policy.
            batch=mf.batch_sample(batch_size=3,num_obs=24,num_regressors=2,max_num_regressors=2,
                                  max_num_categories=2,free_prob=1.,add_interaction=True,
                                  interaction_policy='optional',interaction_prob=0.,link_fun=cfg['link_fn']())
            random=mc.batch_sample(batch_size=3,num_obs=24,num_regressors=2,max_num_regressors=2,
                                   max_num_categories=2,add_interaction=True,
                                   interaction_policy='optional',interaction_prob=0.)
            dataset=ModelClassDataset(dict(families={'DDM':dict(model_cls=cfg['model_cls'],prior_fn=cfg['prior_fn'],
                intrinsic_params=cfg['intrinsic_params'],mask_randomizer_kwargs=mf.mask_randomizer_kwargs)},
                link_funs={'DDM':cfg['link_fn']}),dict(batch_size=3,
                model_family_config=dict(max_num_regressors=2,max_num_categories=2,keep_intercept=True,
                    add_interaction=True,interaction_policy='optional',interaction_prob=0.),
                train_sample_config=dict(num_obs=24,num_regressors=2)),num_workers=0)
            training_batch=next(iter(dataset))
            is_interaction=training_batch['regressor_indices'][:,:,1]>0
            self.assertFalse(training_batch['param_masks'][is_interaction].any())
        for samples in (batch,random):
            masks=samples['param_masks'].reshape(3,-1,samples['n_params'])
            interactions=samples['position_codes'][:,:,1]>0
            self.assertFalse(masks[interactions].any())
        adapted=Adapter.adapt(random,device='cpu')
        model=small_model(input_dim=adapted['input_data'].shape[-1]).train()
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
        predicted,target=model(adapted['param_matrices'][...,None],adapted['input_data'],adapted['param_indices'],
                               adapted['regressor_indices'],adapted['param_masks'],adapted['model_ids'])
        loss=model.compute_loss(predicted,target,adapted['param_masks']);self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.structure_embedding.parameters()),0)
        optimizer.step();model.eval();model._compiled_vel=model.decoder.velocity
        summaries=[]
        hook=model.encoder.register_forward_hook(lambda module,inputs,result:summaries.append(result.detach().clone()))
        with torch.no_grad(),patch('cogformer.networks.transformers.cogformer.tqdm',lambda iterator,**kw:iterator):
            model(adapted['param_matrices'][...,None],adapted['input_data'],adapted['param_indices'],
                  adapted['regressor_indices'],adapted['param_masks'],adapted['model_ids'])
            draws=chunked_sample(model,adapted,num_samples=3,steps=2,solver='rk4',compact=False)
        hook.remove()
        self.assertEqual(draws.shape,(3,3,adapted['param_masks'].shape[1],1))
        self.assertTrue(np.isfinite(draws).all())
        torch.testing.assert_close(summaries[0],summaries[1],rtol=0,atol=0)

    def test_resume_contract_checks_prior_and_encoder(self):
        legacy=structure_training_schema({'add_interaction':True})
        optional=structure_training_schema({'add_interaction':True,'interaction_policy':'optional'})
        conditioned=structure_training_schema({'add_interaction':True},True)
        validate_structure_resume({'state_dict':{}},legacy)
        validate_structure_resume({'structure_schema':optional},optional)
        for requested in (optional,conditioned):
            with self.assertRaises(ValueError):validate_structure_resume({'state_dict':{}},requested)
        with self.assertRaises(ValueError):validate_structure_resume({'structure_schema':optional},legacy)

    def test_checkpoint_writers_preserve_structure_contract(self):
        from experiments.experiment_1 import model_class_cf_train as training
        from experiments.experiment_3.model_class_cf_finetune_lba import LBAFinetuner
        prior=dict(add_interaction=True,interaction_policy='optional',interaction_prob=.5)
        schema=structure_training_schema(prior,True)
        config=dict(epochs=0,steps_per_epoch=1,learning_rate=1e-3,device=torch.device('cpu'),
                    model_family_config=prior,train_sample_config={},batch_size=1)
        with tempfile.TemporaryDirectory() as directory:
            trainer=training.CogFormerTrainer(self.model,None,dict(families={},link_funs={}),Adapter(),num_sim_workers=0)
            with patch.object(training,'CHECKPOINT_DIR',directory),patch.object(trainer,'_prepare_val_data'):
                trainer.train(config,{},checkpoint_path='native.pt')
                saved=torch.load(Path(directory)/'native.pt',weights_only=True)
                self.assertEqual(saved['structure_schema'],schema)
                wrong=dict(config,model_family_config={'add_interaction':True},
                           structure_schema=structure_training_schema({'add_interaction':True},True))
                with self.assertRaises(ValueError):trainer.train(wrong,{},checkpoint_path='native.pt',resume=True)
            finetuner=LBAFinetuner(self.model,None,None,None,Adapter())
            finetuner.train(config,{},checkpoint_dir=directory,checkpoint_path='finetune.pt')
            saved=torch.load(Path(directory)/'finetune.pt',weights_only=True)
            self.assertEqual(saved['structure_schema'],schema)

    def test_conditioned_forward_supports_full_graph_capture(self):
        compiled=torch.compile(self.model,backend='eager',fullgraph=True)
        values=torch.zeros(1,4,1)
        prediction,target=compiled(values,self.x,self.params,self.regs,self.mask,torch.zeros(1,dtype=torch.long))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertEqual(prediction.shape,target.shape)


if __name__=='__main__':unittest.main()
