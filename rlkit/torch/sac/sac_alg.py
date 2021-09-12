import numpy as np
from collections import OrderedDict
import os
import pickle as pkl
import joblib

import torch
import torch.optim as optim
from torch import nn
from torch import autograd
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.torch_base_algorithm import TorchBaseAlgorithm
from rlkit.core import logger, eval_util
from rlkit.samplers import PathSampler
from rlkit.samplers.l2s_sampler import L2SPathSampler



class SACAlgorithm(TorchBaseAlgorithm):
    def __init__(
        self,
        policy_trainer,
        batch_size=1024,
        num_update_loops_per_train_call=1,
        if_save_policy=False,
        policy_save_freq=10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.policy_trainer = policy_trainer
        self.batch_size = batch_size
        self.num_update_loops_per_train_call = num_update_loops_per_train_call
        self.if_save_policy = if_save_policy
        self.policy_save_freq = policy_save_freq


    def get_batch(self, batch_size,keys=None):
        batch = self.replay_buffer.random_batch(batch_size, keys=keys)
        batch = np_to_pytorch_batch(batch)
        return batch


    def _end_epoch(self):
        self.policy_trainer.end_epoch()
        super()._end_epoch()


    def evaluate(self, epoch):
        if self.if_save_policy and epoch%self.policy_save_freq==0 and epoch>0:
            #TODO: save policy
            save_dir = logger.get_curdir()
            save_path = os.path.join(save_dir,"policy_%d.pkl"%epoch)
            joblib.dump(self.policy_trainer.policy,save_path,compress=3)
            save_path = os.path.join(save_dir,"q_%d.pkl"%epoch)
            save_data = {
                'qf1':self.policy_trainer.qf1,
                'qf2':self.policy_trainer.qf2,
                'target_qf1':self.policy_trainer.target_qf1,
                'target_qf2':self.policy_trainer.target_qf2,
            }
            joblib.dump(save_data,save_path,compress=3)
            print("save the policy and qfs")
            
        if self.eval_statistics is None:
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


class DDPGAlgorithm(SACAlgorithm):
    def __init__(self,var,**kwargs):
        super().__init__(**kwargs)
        self.var = var 
    
    def _get_action_and_info(self, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
        action,_ = self.exploration_policy.get_action(observation,deterministic=True)
        action = np.clip(np.random.normal(action, self.var), -1, 1)
        return action,_
    
    def _end_epoch(self):
        self.policy_trainer.end_epoch()
        self.var*=0.9995
        super()._end_epoch()
    
    def evaluate(self, epoch):
        """
        Evaluate the policy, e.g. save/print progress.
        :param epoch:
        :return:
        """
        if self.if_save_policy and epoch%self.policy_save_freq==0 and epoch>0:
            #TODO: save policy
            save_dir = logger.get_curdir()
            save_path = os.path.join(save_dir,"policy_%d.pkl"%epoch)
            joblib.dump(self.policy_trainer.policy,save_path,compress=3)
            print("save the policy")
            
        self.eval_statistics = OrderedDict()
        self.eval_statistics.update(self.policy_trainer.get_eval_statistics())

        statistics = OrderedDict()
        try:
            statistics.update(self.eval_statistics)
            self.eval_statistics = None
        except:
            print('No Stats to Eval')

        logger.log("Collecting samples for evaluation")
        test_paths = self.eval_sampler.obtain_samples()

        statistics.update(eval_util.get_generic_path_information(
            test_paths, stat_prefix="Test",
        ))
        
        average_returns = eval_util.get_average_returns(test_paths)
        statistics['AverageReturn'] = average_returns
        for key, value in statistics.items():
            logger.record_tabular(key, value)
        
        best_statistic = statistics[self.best_key]
        if best_statistic > self.best_statistic_so_far:
            self.best_statistic_so_far = best_statistic
            if self.save_best and epoch >= self.save_best_starting_from_epoch:
                data_to_save = {
                    'epoch': epoch,
                    'statistics': statistics
                }
                data_to_save.update(self.get_epoch_snapshot(epoch))
                logger.save_extra_data(data_to_save, 'best.pkl')
                print('\n\nSAVED BEST\n\n')
    



