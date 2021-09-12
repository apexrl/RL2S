import gym
import os, json, time,heapq,random
import numpy as np
from collections import OrderedDict
import gtimer as gt

from rlkit.data_management.traj_l2s_buffer import TrajL2SBuffer
from rlkit.torch.l2s.dual_env import DualEnv
from rlkit.torch.l2s.learn_env_sac import LearnEnv
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.core import logger, eval_util
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers import PathSampler
from rlkit.samplers.l2s_sampler import L2SPathSampler
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.torch.l2s.l2s_base import L2SBase
from rlkit.torch.l2s.learn_env_sac import LearnEnv
from rlkit.torch.l2s.utils import normalize_acts, normalize_obs
from rlkit.torch.l2s.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.offline_utils import clip_gradient


import torch
import torch.optim as optim
from torch import autograd
import torch.nn.functional as F

class L2SBC(L2SBase):
    def __init__(self, 
        lr,
        momentum,
        train_mode='MLE',
        policy_optim_batch_size=1024,
        num_update_loops_per_train_call=1,
        batch_freq=10,
        **kwargs):
        
        """[summary]

        Args:
            mode ([str]): in ['airl','gail']
            env ([gym.env]): [primal_env] used in LearnEnv
            dual_env_set_train ([policy]): dual_env(i.e. policy), used for training (training policy set)
            dual_env_set_train ([policy]): dual_env(i.e. policy), used for testing (test policy set)
            expert_dataset_list_train ([TrajL2SBuffer]): a list of TrajL2SBuffer corresponds to training policy with primal env
            expert_dataset_list_test ([TrajL2SBuffer]): a list of TrajL2SBuffer corresponds to test policy data with primal env
            gamma (float): [description]. discounted rate
            policy_optim_batch_size[int]: batch size when train policy
            num_update_loops_per_train_call[int]: loops in a train call
            batch_freq (int, optional): [description]. Defaults to 10, frequence to rechoose data.
        """
        super().__init__(**kwargs)
        self.optimizer = torch.optim.Adam(
            self.exploration_policy.parameters(),
            lr=lr,
            betas=(momentum, 0.999)
        )

        self.train_mode = train_mode

        self.policy_optim_batch_size = policy_optim_batch_size
        self.num_update_loops_per_train_call = num_update_loops_per_train_call
        
        self.batch_freq = batch_freq
        self.batch_times = 0
        self.choosed_indexes = None

    def _init_index(self):
        # initialize the self.choosed_indexes
        # if self.choosed_indexes is None or self.batch_times%self.batch_freq==0:
        print('_init_index_begin')
        if self.use_robust:
            self.choosed_indexes,fake_samples = self.compute_diff(is_train=True,policy_num=self.bad_policy_num)
        print('_init_index_done')

    def get_batch(self,batch_size,keys=None):
        expert_data = self.expert_dataset_list_train[self.choosed_cur_index].random_batch(self.policy_optim_batch_size, keys=keys)
        if self.if_normalize_dataset:
            expert_data = self.normalize_batch(expert_data)
        expert_obs = expert_data['observations']
        expert_acts = expert_data['actions']
        
        dual_obs = np.concatenate([expert_obs,expert_acts],axis=-1)
        expert_data['observations'] = dual_obs
        
        if self.use_delta:
            expert_data['actions'] = expert_data['next_observations'] - expert_obs
        else:
            expert_data['actions'] = expert_data['next_observations']
        
        expert_data= np_to_pytorch_batch(expert_data)
        return expert_data

    def compute_diff(self, is_train=True,policy_num=2,get_diff=False):
        if is_train:
            dual_env_set = self.dual_env_set_train
            dual_sampler = self.train_sampler_list
        else:
            dual_env_set = self.dual_env_set_test
            dual_sampler = self.test_sampler_list
        
        real_rwds = self.get_rwd_from_dataset(is_train)
        real_rwds = np.array(real_rwds)
        fake_rwds, fake_samples, steps_list = self.get_stat_from_sampler(dual_sampler,False)
        fake_rwds = np.array(fake_rwds)
        diff = np.abs(real_rwds-fake_rwds)
        diff = list(diff)
        if is_train:
            choosed_index = list(map(diff.index,heapq.nlargest(policy_num,diff)))
        # fake_samples = [fake_samples[i] for i in choosed_index]
        if get_diff:
            #get_diff is deprecated
            return diff,steps_list
        else:
            return choosed_index,fake_samples
 
    def get_both_stat_for_evaluate(self,is_train=False):
        if is_train:
            sampler_list = self.train_sampler_list
        else:
            sampler_list = self.test_sampler_list
        
        fake_rwds,_,step_list = self.get_stat_from_sampler(sampler_list)

        real_rwds = self.get_rwd_from_dataset(is_train)
        real_rwds = np.array(real_rwds)
        fake_rwds = np.array(fake_rwds)
        rwd_diff = np.abs(real_rwds-fake_rwds)
        rwd_diff = list(rwd_diff)
        return rwd_diff, step_list, fake_rwds
            
    def start_training(self, start_epoch=0):

        self._choose_envs()

        # if self.use_robust:
        #     cur_index = random.sample(self.choosed_indexes,1)[0]
        # else:
        #     cur_index = random.sample(len(self.expert_dataset_list_train),1)[0]
        

        for epoch in gt.timed_for(
            range(start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self._start_epoch(epoch)
            for steps_this_epoch in range(self.num_env_steps_per_epoch):
                if self._n_env_steps_total % self.num_steps_between_train_calls == 0:
                    gt.stamp('sample')
                    self._try_to_train(epoch)
                    gt.stamp('train')
                    if self._can_train():
                        self.batch_times+=1
                    if self.use_robust and self.batch_times%self.batch_freq==0:
                        self.choosed_indexes,choosed_samples = self.compute_diff(is_train=True,policy_num=self.bad_policy_num)
                    self._choose_envs() #change to new envs
                    gt.stamp("compute_diff")
            gt.stamp('sample')
            self._try_to_eval(epoch)
            gt.stamp('eval')
            self._end_epoch()

    def _do_training(self,epoch):
        for t in range(self.num_update_loops_per_train_call):
            self._do_policy_training(epoch)

    def _do_policy_training(self,epoch):
        #policy_batch = self.get_batch(self.policy_optim_batch_size, False)
        policy_batch = self.get_batch(self.policy_optim_batch_size)
        obs = policy_batch['observations']
        acts = policy_batch['actions']
        #print(torch.mean(acts,dim=0))
        # print(acts[2])
        # exit(1)

        self.optimizer.zero_grad()
        if self.train_mode=='MLE':
            log_prob = self.exploration_policy.get_log_prob(obs, acts)
            loss = -1.0 * log_prob.mean()
            if self.eval_statistics is None:
                self.eval_statistics = OrderedDict()
                self.eval_statistics['Log-Likelihood'] = ptu.get_numpy(-1.0*loss)
        elif self.train_mode=='MSE':
            pred_acts = self.exploration_policy(obs,deterministic=True)[0]
            #print(pred_acts[0])
            squared_diff = (pred_acts - acts)**2
            loss = torch.sum(squared_diff, dim=-1).mean()
            if self.eval_statistics is None:
                self.eval_statistics = OrderedDict()
                self.eval_statistics['MSE'] = ptu.get_numpy(loss)
        loss.backward()
        clip_gradient(self.optimizer,grad_clip=0.5)
        self.optimizer.step()
   
    def evaluate(self, epoch):
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
        
        rwd_diff, steps_list, rwd_list = self.get_both_stat_for_evaluate(is_train=False)
        for i in range(len(rwd_diff)):
            self.eval_statistics["test_rwd_%d"%i] = rwd_list[i]
            self.eval_statistics["test_rwd_diff_%d"%i] = rwd_diff[i]
            self.eval_statistics["test_ep_step_%d"%i] = steps_list[i]
        self.eval_statistics["test_rwd_diff_min"] = min(rwd_diff)
        self.eval_statistics["test_rwd_diff_max"] = max(rwd_diff)
        self.eval_statistics["test_rwd_diff_mean"] = np.mean(rwd_diff)
        self.eval_statistics["test_rwd_diff_std"] = np.std(rwd_diff)
        self.eval_statistics["test_rwd_min"] = min(rwd_list)
        self.eval_statistics["test_rwd_max"] = max(rwd_list)
        self.eval_statistics["test_rwd_mean"] = np.mean(rwd_list)
        self.eval_statistics["test_rwd_std"] = np.std(rwd_list)

        rwd_diff, steps_list, rwd_list = self.get_both_stat_for_evaluate(is_train=True)
        for i in range(len(rwd_diff)):
            self.eval_statistics["train_rwd_%d"%i] = rwd_list[i]
            self.eval_statistics["train_rwd_diff_%d"%i] = rwd_diff[i]
            self.eval_statistics["train_ep_step_%d"%i] = steps_list[i]
        self.eval_statistics["train_rwd_diff_min"] = min(rwd_diff)
        self.eval_statistics["train_rwd_diff_max"] = max(rwd_diff)
        self.eval_statistics["train_rwd_diff_mean"] = np.mean(rwd_diff)
        self.eval_statistics["train_rwd_diff_std"] = np.std(rwd_diff)
        self.eval_statistics["train_rwd_min"] = min(rwd_list)
        self.eval_statistics["train_rwd_max"] = max(rwd_list)
        self.eval_statistics["train_rwd_mean"] = np.mean(rwd_list)
        self.eval_statistics["train_rwd_std"] = np.std(rwd_list)
        
        super().evaluate(epoch)
    
    def _choose_envs(self):
        if self.use_robust:
            self.choosed_cur_index = random.sample(self.choosed_indexes,1)[0]
        else:
            self.choosed_cur_index = random.sample(list(np.arange(len(self.expert_dataset_list_train))),1)[0]

    @property
    def networks(self):
        return [self.exploration_policy]
        #+ [i.model for i in self.dual_env_set_train] + [i.model for i in self.dual_env_set_test]
    
    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)        
        return snapshot

    def to(self,device):
        for i in self.dual_env_set_train:
            i.model.to(device)
        for i in self.dual_env_set_test:
            i.model.to(device)

        super().to(device)

    def _can_evaluate(self):
        return True

    def _can_train(self):
        return True


    def test_batch(self,keys=['observations','next_observations','actions']):
        # if self.use_robust:
        #     if self.choosed_indexes is None or self.batch_times%self.batch_freq==0:
        #         self.choosed_indexes,self.choosed_samples = self.compute_diff(is_train=True,policy_num=self.bad_policy_num)
        #     cur_index = random.sample(self.choosed_indexes,1)[0]
        # else:
        #     cur_index = random.sample(len(self.expert_dataset_list_train),1)[0]
        
        expert_data = self.expert_dataset_list_train[0]._get_batch_using_indices(np.arange(0,2), keys=keys)
        print(expert_data['observations'][0])

        if self.if_normalize_dataset:
            expert_data = self.normalize_batch(expert_data)
        expert_obs = expert_data['observations']
        expert_acts = expert_data['actions']
        print(expert_obs[0])

        dual_obs = np.concatenate([expert_obs,expert_acts],axis=-1)
        expert_data['observations'] = dual_obs
        
        if self.use_delta:
            expert_data['actions'] = expert_data['next_observations'] - expert_obs
        else:
            expert_data['actions'] = expert_data['next_observations']
        
        print(expert_data['observations'][0])
        print(expert_obs[0])
        expert_data = self.expert_dataset_list_train[0]._get_batch_using_indices(np.arange(0,2), keys=keys)
        print(expert_data['observations'][0])
        exit(1)