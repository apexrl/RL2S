import numpy as np
from collections import OrderedDict
import os
import pickle as pkl
import joblib
import gtimer as gt

import torch
import torch.optim as optim
from torch import nn
from torch import autograd
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.torch_base_algorithm import TorchBaseAlgorithm
from rlkit.torch.sac.sac_alg import SACAlgorithm
from rlkit.data_management.path_builder import PathBuilder
from rlkit.core import logger, eval_util
from rlkit.samplers import PathSampler
from rlkit.samplers.l2s_sampler import L2SPathSampler


class DownstreamSACAlgorithm(TorchBaseAlgorithm):
    def __init__(
        self,
        policy_trainer,
        max_rollout_length,
        env_state_buffer,
        if_set_state=True,
        batch_size=1024,
        num_update_loops_per_train_call=1,
        if_save_policy=False,
        policy_save_freq=10,
        **kwargs
    ):
        """[summary]

        Args:
            policy_trainer ([type]): [description]
            rollout_length ([type]): [description]
            env_state_buffer ([List]): True env state
            if_set_state (bool, optional): [description]. Defaults to True.
            batch_size (int, optional): [description]. Defaults to 1024.
            num_update_loops_per_train_call (int, optional): [description]. Defaults to 1.
            if_save_policy (bool, optional): [description]. Defaults to False.
            policy_save_freq (int, optional): [description]. Defaults to 10.
        """
        super().__init__(**kwargs)
        self.policy_trainer = policy_trainer
        self.batch_size = batch_size
        self.num_update_loops_per_train_call = num_update_loops_per_train_call
        self.if_save_policy = if_save_policy
        self.policy_save_freq = policy_save_freq
        self.rollout_length = 1
        self.max_rollout_length = max_rollout_length
        self.env_state_buffer = env_state_buffer
        self.if_set_state = if_set_state
        self.env_buffer_size = len(self.env_state_buffer)
    
    def start_training(self, start_epoch=0):
        self._current_path_builder = PathBuilder()
        observation = self._start_new_rollout()

        for epoch in gt.timed_for(
            range(start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self._start_epoch(epoch)
            for steps_this_epoch in range(self.num_env_steps_per_epoch):
                action, agent_info = self._get_action_and_info(observation)
                if self.render: self.training_env.render()

                next_ob, raw_reward, terminal, env_info = (
                    self.training_env.step(action)
                )
                if self.no_terminal: terminal = False
                self._n_env_steps_total += 1

                reward = np.array([raw_reward])
                terminal = np.array([terminal])
                self._handle_step(
                    observation,
                    action,
                    reward,
                    next_ob,
                    np.array([False]) if self.no_terminal else terminal,
                    absorbing=np.array([0., 0.]),
                    agent_info=agent_info,
                    env_info=env_info,
                )
                if terminal[0]:
                    if self.wrap_absorbing:
                        raise NotImplementedError()
                    self._handle_rollout_ending()
                    observation = self._start_new_rollout()
                    self._set_rollout_length(epoch)
                elif len(self._current_path_builder) >= self.rollout_length:
                    self._handle_rollout_ending()
                    observation = self._start_new_rollout()
                    self._set_rollout_length(epoch)
                else:
                    observation = next_ob

                if self._n_env_steps_total % self.num_steps_between_train_calls == 0:
                    gt.stamp('sample')
                    self._try_to_train(epoch)
                    gt.stamp('train')

            gt.stamp('sample')
            self._try_to_eval(epoch)
            gt.stamp('eval')
            self._end_epoch()

    
    def get_batch(self, batch_size,keys=None):
        batch = self.replay_buffer.random_batch(batch_size, keys=keys)
        batch = np_to_pytorch_batch(batch)
        return batch


    def _end_epoch(self):
        self.policy_trainer.end_epoch()
        super()._end_epoch()

    def _start_new_rollout(self):
        self.exploration_policy.reset()
        if self.if_set_state:
            sampled_index = np.random.choice(self.env_buffer_size,1)[0]
            sampled_data = self.env_state_buffer[sampled_index]
            new_state = sampled_data
            new_state = self.training_env.set_state(new_state)
        else:
            new_state = self.training_env.reset()
        return new_state

    def _set_rollout_length(self,epoch):
        delta = int(epoch * 0.01)
        self.rollout_length=1+delta
        self.rollout_length=min(self.rollout_length,self.max_rollout_length)

    def evaluate(self, epoch):
        if self.if_save_policy and epoch%self.policy_save_freq==0 and epoch>0:
            #TODO: save policy
            save_dir = logger.get_curdir()
            save_path = os.path.join(save_dir,"policy_%d.pkl"%epoch)
            joblib.dump(self.policy_trainer.policy,save_path,compress=3)
            print("save the policy")
            
        self.eval_statistics = OrderedDict()
        self.eval_statistics.update(self.policy_trainer.get_eval_statistics())
        super().evaluate(epoch)

    def _do_training(self, epoch):
        for t in range(self.num_update_loops_per_train_call):
            self._do_policy_training(epoch)

    def _do_policy_training(self, epoch):
        policy_batch = self.get_batch(self.batch_size)
        self.policy_trainer.train_step(policy_batch)  

    def _can_evaluate(self):
        return self.replay_buffer.num_steps_can_sample() >= self.min_steps_before_training 
    
    
    @property
    def networks(self):
        return self.policy_trainer.networks


    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(self.policy_trainer.get_snapshot())
        return snapshot


    def to(self, device):
        super().to(device)


class DownstreamTestedSACAlgorithm(SACAlgorithm):
    def __init__(
        self,
        test_env,
        **kwargs
    ): 
        super().__init__(**kwargs)

        self.learned_env = test_env
        self.learned_eval_sampler = eval_sampler = PathSampler(
                self.learned_env,
                self.eval_policy,
                self.num_steps_per_eval,
                self.max_path_length,
                no_terminal=self.no_terminal,
                render=self.render,
            )

    
    def evaluate(self, epoch):
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
        learned_test_paths = self.learned_eval_sampler.obtain_samples()
        average_returns = eval_util.get_average_returns(learned_test_paths)
        self.eval_statistics['AverageReturn Fake'] = average_returns
        super().evaluate(epoch)