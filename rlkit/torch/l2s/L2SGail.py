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
from rlkit.torch.l2s.policies import MakeDeterministic
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer,L2SEnvReplayBuffer
from rlkit.torch.l2s.l2s_base import L2SBase
from rlkit.torch.l2s.learn_env_sac import LearnEnv
from rlkit.torch.l2s.utils import normalize_acts, normalize_obs
from rlkit.torch.l2s.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.offline_utils import clip_gradient


import torch
import torch.optim as optim
from torch import autograd
import torch.nn.functional as F

class L2SGail(L2SBase):
    def __init__(self, 
        mode,
        policy_trainer,
        if_set_state=False,

        disc_optim_batch_size=1024,
        policy_optim_batch_size=1024,

        num_update_loops_per_train_call=1,
        num_disc_updates_per_loop_iter=100,
        num_policy_updates_per_loop_iter=100,

        replay_buffer = None,
        replay_buffer_size = None,

        disc_lr=1e-3,
        disc_l2_coef = 0,
        disc_momentum=0.0,
        disc_optimizer_class=optim.Adam,

        rew_clip_min=None,
        rew_clip_max=None,

        is_disc=False,
        batch_freq=10,

        **kwargs):
        
        """[summary]

        Args:
            mode ([str]): in ['airl','gail']
            policy_trainer ([Trainer]): [generator]
            discriminator ([Trainer]): [discriminator]
            env ([gym.env]): [primal_env] used in LearnEnv
            dual_env_set_train ([policy]): dual_env(i.e. policy), used for training (training policy set)
            dual_env_set_train ([policy]): dual_env(i.e. policy), used for testing (test policy set)
            expert_dataset_list_train ([TrajL2SBuffer]): a list of TrajL2SBuffer corresponds to training policy with primal env
            expert_dataset_list_test ([TrajL2SBuffer]): a list of TrajL2SBuffer corresponds to test policy data with primal env
            gamma (float): [description]. discounted rate
            disc_optim_batch_size[int]: batch size when train discriminator
            policy_optim_batch_size[int]: batch size when train policy
            num_update_loops_per_train_call[int]: loops in a train call
            num_disc_updates_per_loop_iter[int]: update steps in a loop of train call
            num_policy_updates_per_loop_iter[int]: update steps in a loop of train call
            is_disc [bool]: use occupancy discrenpancy to choose data if True, reward discrepancy otherwise
            batch_freq (int, optional): [description]. Defaults to 10, frequence to rechoose data.
        """
        super().__init__(**kwargs)
        self.mode=mode
        self.policy_trainer = policy_trainer
        self.if_set_state = if_set_state
        
        # eval_policy = MakeDeterministic(self.model)

        # self.discriminator = discriminator
        self.disc_optim_batch_size = disc_optim_batch_size
        self.policy_optim_batch_size = policy_optim_batch_size

        self.num_update_loops_per_train_call = num_update_loops_per_train_call
        self.num_disc_updates_per_loop_iter = num_disc_updates_per_loop_iter
        self.num_policy_updates_per_loop_iter = num_policy_updates_per_loop_iter

        self.replay_buffer_list = []
        self.replay_buffer_size = replay_buffer_size
        
        if replay_buffer is None:
            # assert self.max_path_length < replay_buffer_size
            for i in range(len(self.dual_env_set_train)):
                #dual_env = self.dual_env_set_train[i]
                replay_buffer = L2SEnvReplayBuffer(
                    self.replay_buffer_size,
                    self.dual_env_set_train[0],
                    random_seed = np.random.randint(10000)
                )
                self.replay_buffer_list.append(replay_buffer)

        self.disc_lr = disc_lr
        self.disc_l2_coef = disc_l2_coef
        self.disc_momentum = disc_momentum
        self.disc_optimizer = disc_optimizer_class(
            self.discriminator.parameters(),
            lr=disc_lr,
            betas=(disc_momentum, 0.999),
            weight_decay=self.disc_l2_coef,
        )


        self.rew_clip_min = rew_clip_min
        self.rew_clip_max = rew_clip_max
        self.clip_min_rews = rew_clip_min is not None
        self.clip_max_rews = rew_clip_max is not None

        self.disc_eval_statistics = None

        self.is_disc = is_disc
        
        self.batch_freq = batch_freq
        self.batch_times = 0
        self.choosed_indexes = None

        self.logger_reward_expert_D=[]
        self.logger_reward_theta_D=[]
        self.train_diff_record = []
        self.train_ep_steps_record = []
        self.disc_loss_record = []

    def _init_index(self):
        #initialize the self.choosed_indexes
        #if self.choosed_indexes is None or self.batch_times%self.batch_freq==0:
        if self.use_robust:
            self.choosed_indexes,fake_samples = self.compute_diff(is_train=True,is_disc=self.is_disc,policy_num=self.bad_policy_num)
        self._choose_envs()
        print("_init_index_done")

    def get_batch(self,batch_size,train_reward=False,keys=None):
        #concat the data
        if train_reward:
            expert_data = self.expert_dataset_list_train[self.choosed_cur_index].random_batch(batch_size, keys=keys)
            policy_data = self.replay_buffer_list[self.choosed_cur_index].random_batch(batch_size, keys=keys)
            if self.if_normalize_dataset:
                expert_data = self.normalize_batch(expert_data)
            expert_obs = expert_data['observations']
            expert_acts = expert_data['actions']

            # if self.if_normalize_obs:
            #     expert_obs = normalize_obs(expert_obs,self.env_observation_mean,self.env_observation_std)
            # if self.if_normalize_acts:
            #     expert_acts = normalize_acts(expert_acts,self.env_action_mean,self.env_action_std)
            

            dual_obs = np.concatenate([expert_obs,expert_acts],axis=-1)
            expert_data['observations'] = dual_obs
            
            expert_data['actions'] = expert_data['next_observations']

            # if self.use_delta:
            #     expert_data['actions'] = expert_data['next_observations'] - expert_obs
            #     policy_data['actions'] = policy_data['actions'] - policy_data['observations'][:,:self.dual_action_dim]
            # else:
            #     expert_data['actions'] = expert_data['next_observations']
            
            expert_data= np_to_pytorch_batch(expert_data)
            policy_data= np_to_pytorch_batch(policy_data)
            #self.batch_times+=1
            return expert_data,policy_data
        else:
            policy_data = self.replay_buffer_list[self.choosed_cur_index].random_batch(batch_size, keys=keys)
            policy_data= np_to_pytorch_batch(policy_data)
            return policy_data

    def compute_diff(self, is_train=True,is_disc=False,policy_num=2,get_diff=False):
        if is_train:
            dual_env_set = self.dual_env_set_train
            dual_sampler = self.train_sampler_list
            expert_traj_nums = self.expert_traj_nums_train
        else:
            dual_env_set = self.dual_env_set_test
            dual_sampler = self.test_sampler_list
            expert_traj_nums = self.expert_traj_nums_test
        
        if is_disc:
            real_disc = self.get_disc_from_dataset(is_train)
            real_disc = np.array(real_disc)/expert_traj_nums
            fake_disc, fake_samples, steps_list = self.get_stat_from_sampler(dual_sampler,True)
            fake_disc = np.array(fake_disc)
            diff = np.abs(real_disc-fake_disc)
        else:
            real_rwds = self.get_rwd_from_dataset(is_train)
            real_rwds = np.array(real_rwds)
            fake_rwds, fake_samples, steps_list = self.get_stat_from_sampler(dual_sampler,False)
            fake_rwds = np.array(fake_rwds)
            diff = np.abs(real_rwds-fake_rwds)
        
        diff = list(diff)
        if is_train:
            self.add_sampler_data_to_replay_buffers(fake_samples)
            self.train_diff_record.append(diff)
            self.train_ep_steps_record.append(steps_list)
            choosed_index = list(map(diff.index,heapq.nlargest(policy_num,diff)))
        # fake_samples = [fake_samples[i] for i in choosed_index]
        if get_diff:
            return diff,steps_list
        else:
            return choosed_index,fake_samples

    def get_disc_from_dataset(self,is_train):
        if is_train:
            buffer_list = self.expert_dataset_list_train
        else:
            buffer_list = self.expert_dataset_list_test
        disc_list = self.get_stat_from_buffer_list(buffer_list,True)
        return disc_list
    
    def get_both_stat_for_evaluate(self,is_train=False):
        if is_train:
            sampler_list = self.train_sampler_list
            expert_traj_nums = self.expert_traj_nums_train
        else:
            sampler_list = self.test_sampler_list
            expert_traj_nums = self.expert_traj_nums_test
        fake_rwds = []
        fake_disc = []
        all_collect_data = []
        step_list = []
        for i in range(len(sampler_list)):
            single_sampler = sampler_list[i]
            collect_paths, rwds, disc, steps = single_sampler.obtain_samples()
            all_collect_data.append(collect_paths)
            step_list.append(steps)
            fake_disc.append(disc)
            fake_rwds.append(rwds)
        real_disc = self.get_disc_from_dataset(is_train)
        real_rwds = self.get_rwd_from_dataset(is_train)
        real_disc = np.array(real_disc)/expert_traj_nums
        real_rwds = np.array(real_rwds)
        fake_rwds = np.array(fake_rwds)
        fake_disc = np.array(fake_disc)
        rwd_diff = np.abs(real_rwds-fake_rwds)
        disc_diff = np.abs(real_disc-fake_disc)
        rwd_diff = list(rwd_diff)
        disc_diff = list(disc_diff)
        fake_rwds = list(fake_rwds)
        return rwd_diff,disc_diff,step_list,fake_rwds
            
    def get_disc_from_buffer(self,buffer):
        """
            compute the disc for expert dataset
        """
        tot_sum = 0.0
        last_index = 0
        #data_size = buffer._buffer_size
        data_size = buffer._raw_buffer_size
        iter_nums = int(data_size/self.policy_optim_batch_size)
        keys = ['cur_step','observations','actions','next_observations']
        tot_sum=None
        #print("iter_nums",iter_nums)
        with torch.no_grad():
            for i in range(iter_nums):
                batch = buffer._get_raw_batch_using_indices(np.arange(last_index,last_index+self.policy_optim_batch_size), keys)
                # buffer is the expert dataset
                if self.if_normalize_dataset:
                    batch = self.normalize_batch(batch)
                #batch = buffer._get_batch_using_indices(np.arange(last_index,last_index+self.policy_optim_batch_size), keys)

                cur_step = np.squeeze(np.array(batch['cur_step']))
                gamma_coef = np.power(self.gamma,cur_step)
                batch['gamma_coef'] = gamma_coef
                batch = np_to_pytorch_batch(batch)
                obs = batch['observations']
                acts = batch['actions']
                next_obs = batch['next_observations']
                gamma_coef = batch['gamma_coef'] #[batch]
                sa_input = torch.cat([obs, acts,next_obs], dim=1)
                unc_disc_value,disc_value = self.discriminator(sa_input)
                disc_value = torch.squeeze(disc_value) # [256]
                cur_sum = torch.sum(disc_value*gamma_coef)
                last_index+=self.policy_optim_batch_size
                if tot_sum is None:
                    tot_sum =cur_sum
                else:
                    tot_sum+=cur_sum
            indices = np.arange(last_index,data_size)
            #batch = buffer._get_batch_using_indices(indices,keys)
            batch = buffer._get_raw_batch_using_indices(indices,keys)
            if self.if_normalize_dataset:
                batch = self.normalize_batch(batch)

            cur_step = np.squeeze(np.array(batch['cur_step']))
            gamma_coef = np.power(self.gamma,cur_step)
            batch['gamma_coef'] = gamma_coef
            batch = np_to_pytorch_batch(batch)
            obs = batch['observations']
            acts = batch['actions']
            next_obs = batch['next_observations']
            gamma_coef = batch['gamma_coef'] #[batch]
            sa_input = torch.cat([obs, acts,next_obs], dim=1)
            unc_disc_value,disc_value = self.discriminator(sa_input)
            disc_value = torch.squeeze(disc_value)
            cur_sum = torch.sum(disc_value*gamma_coef)
            if tot_sum is None:
                tot_sum =cur_sum
            else:
                tot_sum+=cur_sum
        #print(tot_sum)
        #print(type(tot_sum))
        tot_sum = tot_sum.detach().cpu().numpy()
        tot_sum = float(tot_sum)
        return tot_sum

    def start_training(self, start_epoch=0):
        self._current_path_builder = PathBuilder()
        self._choose_envs()
        observation = self._start_new_rollout()

        for epoch in gt.timed_for(
            range(start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self._start_epoch(epoch)
            for steps_this_epoch in range(self.num_env_steps_per_epoch):
                action, agent_info = self._get_action_and_info(observation) # we should choose actions
                if self.render: self.choosed_env.render()

                next_ob, raw_reward, terminal, env_info = (
                    self.choosed_env.step(action)
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
                    # self._handle_rollout_ending()
                    self._current_path_builder = PathBuilder()
                    observation = self._start_new_rollout()
                elif len(self._current_path_builder) >= self.max_path_length:
                    # self._handle_rollout_ending()
                    self._current_path_builder = PathBuilder()
                    observation = self._start_new_rollout()
                else:
                    observation = next_ob

                if self._n_env_steps_total % self.num_steps_between_train_calls == 0:
                    gt.stamp('sample')
                    self._try_to_train(epoch)
                    gt.stamp('train')
                    if self._can_train():
                        self.batch_times+=1
                    if self.use_robust and self.batch_times%self.batch_freq==0:
                        self.choosed_indexes,self.choosed_samples = self.compute_diff(is_train=True,is_disc=self.is_disc,policy_num=self.bad_policy_num)
                    self._choose_envs() #change to new envs
                    observation = self._start_new_rollout()
                    gt.stamp("compute_diff")

            gt.stamp('sample')
            self._try_to_eval(epoch)
            gt.stamp('eval')
            self._end_epoch()

    def _do_training(self,epoch):
        for t in range(self.num_update_loops_per_train_call):
            if t>0 and (t+1)%5==0:
                self._choose_envs()
                while not self._can_train():
                    self._choose_envs()
            for _ in range(self.num_disc_updates_per_loop_iter):
                self._do_reward_training(epoch)
            for _ in range(self.num_policy_updates_per_loop_iter):
                self._do_policy_training(epoch)

    def _do_reward_training(self,epoch):
        '''
            Train the discriminator
        '''
        #self.disc_optimizer.zero_grad()
        keys = ['observations','actions','next_observations']

        # expert_batch = self.get_batch(self.disc_optim_batch_size, True, keys)
        # policy_batch = self.get_batch(self.disc_optim_batch_size, False, keys)
        expert_batch, policy_batch = self.get_batch(self.disc_optim_batch_size,True,keys)
        
        expert_obs = expert_batch['observations']
        expert_acts = expert_batch['actions']
        policy_obs = policy_batch['observations']
        policy_acts = policy_batch['actions']

        policy_acts = self.policy_trainer.policy(policy_obs, return_log_prob=True)[0]
        self._reward_training_step(policy_obs,policy_acts,expert_obs,expert_acts) #

    def _reward_training_step(self,policy_obs,policy_acts,expert_obs,expert_acts):
        self.disc_optimizer.zero_grad()
        expert_disc_input = torch.cat([expert_obs, expert_acts], dim=1)
        policy_disc_input = torch.cat([policy_obs, policy_acts], dim=1)
        disc_input = torch.cat([expert_disc_input, policy_disc_input], dim=0)

        unclamp_logits,disc_logits = self.discriminator(disc_input) # [512,1]

        if torch.any(torch.isnan(unclamp_logits)) or torch.any(torch.isnan(disc_logits)):
            print(torch.mean(expert_disc_input,dim=0))
            print(torch.mean(policy_disc_input,dim=0))

            #print("NAN in do_reward_training")
            self.show_para(self.discriminator.parameters())
            print('-'*50)
            print('-'*50)
            self.show_para(self.policy_trainer.policy.parameters())
            print('-'*50)
            print('-'*50)
            #self.show_para(self.policy_trainer.qf1.parameters())
            exit(1)

        expert_data, policy_data  = disc_logits.chunk(2,0)
        unclamp_expert_data,unclamp_policy_data = unclamp_logits.chunk(2,0)
        self.logger_reward_expert_D.append(np.mean(unclamp_expert_data.detach().cpu().numpy()))
        self.logger_reward_theta_D.append(np.mean(unclamp_policy_data.detach().cpu().numpy()))

        disc_ce_loss = -torch.mean(expert_data)+torch.mean(policy_data)

        disc_grad_pen_loss = 0.0

        self.disc_loss_record.append(disc_ce_loss.detach().cpu().numpy().mean())
        disc_total_loss = disc_ce_loss + disc_grad_pen_loss
        disc_total_loss.backward()
        clip_gradient(self.disc_optimizer,0.5) #clip_grad change from 0.5 --> 0.1
        self.disc_optimizer.step()

        """
        Save some statistics for eval
        """
        if self.disc_eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.disc_eval_statistics = OrderedDict()
            
            self.disc_eval_statistics['Disc CE Loss'] = np.mean(ptu.get_numpy(disc_ce_loss))

    def _do_policy_training(self,epoch):
        #policy_batch = self.get_batch(self.policy_optim_batch_size, False)
        expert_batch, policy_batch = self.get_batch(self.disc_optim_batch_size,train_reward=True)
        self.policy_trainer.train_step(expert_batch)
        self.policy_trainer.train_step(policy_batch)

    def evaluate(self, epoch):
        self.eval_statistics = OrderedDict()
        #if self.disc_eval_statistics is not None:
        self.eval_statistics.update(self.disc_eval_statistics)
        self.eval_statistics.update(self.policy_trainer.get_eval_statistics())

        self.eval_statistics["disc_loss"] = np.mean(self.disc_loss_record)
        self.disc_loss_record = []

        #evaluate test dataset
        rwd_diff, disc_diff, steps_list,fake_rwds = self.get_both_stat_for_evaluate(is_train=False)
        for i in range(len(disc_diff)):
            self.eval_statistics["test_rwd_%d"%i] = fake_rwds[i]
            self.eval_statistics["test_disc_diff_%d"%i] = disc_diff[i]
            self.eval_statistics["test_rwd_diff_%d"%i] = rwd_diff[i]
            self.eval_statistics["test_ep_step_%d"%i] = steps_list[i]
        self.eval_statistics["test_steps_diff_min"] = min(steps_list)
        self.eval_statistics["test_steps_diff_max"] = max(steps_list)
        self.eval_statistics["test_steps_diff_mean"] = np.mean(steps_list)
        self.eval_statistics["test_steps_diff_std"] = np.std(steps_list)
        self.eval_statistics["test_disc_diff_min"] = min(disc_diff)
        self.eval_statistics["test_disc_diff_max"] = max(disc_diff)
        self.eval_statistics["test_disc_diff_mean"] = np.mean(disc_diff)
        self.eval_statistics["test_disc_diff_std"] = np.std(disc_diff)
        self.eval_statistics["test_rwd_diff_min"] = min(rwd_diff)
        self.eval_statistics["test_rwd_diff_max"] = max(rwd_diff)
        self.eval_statistics["test_rwd_diff_mean"] = np.mean(rwd_diff)
        self.eval_statistics["test_rwd_diff_std"] = np.std(rwd_diff)
        self.eval_statistics["test_rwd_min"] = min(fake_rwds)
        self.eval_statistics["test_rwd_max"] = max(fake_rwds)
        self.eval_statistics["test_rwd_mean"] = np.mean(fake_rwds)
        self.eval_statistics["test_rwd_std"] = np.std(fake_rwds)

        #evaluate train dataset
        rwd_diff, disc_diff, steps_list, fake_rwds = self.get_both_stat_for_evaluate(is_train=True)
        for i in range(len(disc_diff)):
            self.eval_statistics["train_rwd_%d"%i] = fake_rwds[i]
            self.eval_statistics["train_disc_diff_%d"%i] = disc_diff[i]
            self.eval_statistics["train_rwd_diff_%d"%i] = rwd_diff[i]
            self.eval_statistics["train_ep_step_%d"%i] = steps_list[i]
        self.eval_statistics["train_steps_diff_min"] = min(steps_list)
        self.eval_statistics["train_steps_diff_max"] = max(steps_list)
        self.eval_statistics["train_steps_diff_mean"] = np.mean(steps_list)
        self.eval_statistics["train_steps_diff_std"] = np.std(steps_list)
        self.eval_statistics["train_disc_diff_min"] = min(disc_diff)
        self.eval_statistics["train_disc_diff_max"] = max(disc_diff)
        self.eval_statistics["train_disc_diff_mean"] = np.mean(disc_diff)
        self.eval_statistics["train_disc_diff_std"] = np.std(disc_diff)
        self.eval_statistics["train_rwd_diff_min"] = min(rwd_diff)
        self.eval_statistics["train_rwd_diff_max"] = max(rwd_diff)
        self.eval_statistics["train_rwd_diff_mean"] = np.mean(rwd_diff)
        self.eval_statistics["train_rwd_diff_std"] = np.std(rwd_diff)
        self.eval_statistics["train_rwd_min"] = min(fake_rwds)
        self.eval_statistics["train_rwd_max"] = max(fake_rwds)
        self.eval_statistics["train_rwd_mean"] = np.mean(fake_rwds)
        self.eval_statistics["train_rwd_std"] = np.std(fake_rwds)

        self.train_diff_record = []
        self.train_ep_steps_record = []
        logger.record_tabular("Policy D in Reward",np.mean(self.logger_reward_theta_D).mean())
        logger.record_tabular("Expert D in Reward",np.mean(self.logger_reward_expert_D).mean())
        self.logger_reward_expert_D=[]
        self.logger_reward_theta_D=[]

        super().evaluate(epoch)
    
    def _end_epoch(self):
        self.policy_trainer.end_epoch()
        self.disc_eval_statistics = None
        super()._end_epoch()
    
    def _start_new_rollout(self):
        self.exploration_policy.reset()
        if self.if_set_state:
            sampled_data = self.expert_dataset_list_train[self.choosed_cur_index].random_batch(1,keys=['observations','actions','next_observations'])
            if self.if_normalize_dataset:
                sampled_data = self.normalize_batch(sampled_data)
            ob = sampled_data['observations'][0]
            act = sampled_data['actions'][0]

            new_state = np.concatenate((ob,act),axis=-1)
            self.choosed_env.steps = 0 
        else:
            new_state = self.choosed_env.reset()
        return new_state
    
    def _handle_step(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminal,
        absorbing,
        agent_info,
        env_info,
    ):
        """
        Implement anything that needs to happen after every step
        :return:
        """
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            absorbing=absorbing,
            agent_infos=agent_info,
            env_infos=env_info,
        )
        self.choosed_replay_buffer.add_sample(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            absorbing=absorbing,
            agent_info=agent_info,
            env_info=env_info,
        )
    
    def _choose_envs(self):
        self._current_path_builder = PathBuilder()
        if self.use_robust:
            self.choosed_cur_index = random.sample(self.choosed_indexes,1)[0]
        else:
            self.choosed_cur_index = random.sample(list(np.arange(len(self.expert_dataset_list_train))),1)[0]
            #while not self.replay_buffer_list[self.choosed_cur_index].num_steps_can_sample() >= self.min_steps_before_training:
                #self.choosed_cur_index = random.sample(list(np.arange(len(self.expert_dataset_list_train))),1)[0]
        self.choosed_env = self.dual_env_set_train[self.choosed_cur_index]
        self.choosed_replay_buffer = self.replay_buffer_list[self.choosed_cur_index]

    def _get_action_and_info(self, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
        action, agent_info = self.exploration_policy.get_action(observation)
        if np.any(np.isnan(action)):
            print("take nan action")
            print(action)
            print('-'*50)
            print(observation)
            print('-'*50)
            self.show_all_para(self.exploration_policy.parameters())
            exit(1)
        if self.use_delta:
            action = action + observation[:self.dual_action_dim]
        
        return action, agent_info
    
    @property
    def networks(self):
        return [self.discriminator]+self.policy_trainer.networks

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(disc=self.discriminator)
        snapshot.update(self.policy_trainer.get_snapshot())
        return snapshot

    def to(self,device):
        for i in self.dual_env_set_train:
            i.model.to(device)
        for i in self.dual_env_set_test:
            i.model.to(device)
        super().to(device)

    def _can_evaluate(self):
        return self._can_train() and self.disc_eval_statistics is not None and self.disc_eval_statistics

    def _can_train(self):
        return self.choosed_replay_buffer.num_steps_can_sample() >= self.min_steps_before_training

    def sample_data(self,num_steps):
        collect_paths, rwds, disc, steps = self.train_sampler_list[self.choosed_cur_index].obtain_samples(num_steps)
        pass

    def add_sampler_data_to_replay_buffers(self,data):
        for index in range(len(self.replay_buffer_list)):
            trajs = data[index]
            for traj in trajs:
                obs=np.array(traj["observations"])
                acts=np.array(traj["actions"])
                rwds=np.array(traj["rewards"])
                next_obs=np.array(traj["next_observations"])
                terminal=np.array(traj["terminals"])
                agent_info=np.array(traj["agent_info"])
                env_info = np.array(traj["env_info"])
                absorbing = np.array(traj['absorbing'])

                dual_obs = np.concatenate((obs[:-1],acts[:-1]),axis=-1)
                dual_acts = next_obs[:-1]
                dual_next_obs = np.concatenate((next_obs[:-1],acts[1:]),axis=-1)
                dual_rwds = rwds[:-1]
                dual_terminal=terminal[:-1]
                dual_absorbing = absorbing[:-1]
                dual_agent_info=agent_info[:-1]
                dual_env_info = env_info[:-1]

                for (
                    ob,
                    action,
                    next_ob,
                    reward,
                    terminal,
                    absorbing,
                    agent_info,
                    env_info
                ) in zip(
                    dual_obs,
                    dual_acts,
                    dual_next_obs,
                    dual_rwds,
                    dual_terminal,
                    dual_absorbing,
                    dual_agent_info,
                    dual_env_info,
                ):
                    self.replay_buffer_list[index].add_sample(
                        observation=ob,
                        action=action,
                        next_observation=next_ob,
                        reward=reward,
                        terminal=terminal,
                        absorbing=absorbing,
                        agent_info=agent_info,
                        env_info=env_info,
                    )
