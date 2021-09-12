from collections import defaultdict
import random as python_random
from random import sample
from itertools import starmap
from functools import partial

import numpy as np

from rlkit.data_management.replay_buffer import ReplayBuffer

class TrajL2SBuffer():
    '''
        THE MAX LENGTH OF AN EPISODE SHOULD BE STRICTLY SMALLER THAN THE max_replay_buffer_size
        OTHERWISE THERE IS A BUG IN TERMINATE_EPISODE

        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
    '''
    def __init__(
        self,
        trajectories,
        random_seed=1995,
        gamma=0.99,
        action_dim = None,
        buffer_limit=None,
        traj_lim=100,
    ):
        self.gamma = gamma
        self._action_dim = action_dim
        self.traj_lim = traj_lim
        self._actions=[]
        self._rewards=[]
        self._terminals=[]
        self._observations=[]
        self._next_obs=[]
        self._cur_step = []

        self._gamma_remain = []
        self._future_reward = []
        self._last_observation = []
        self._np_rand_state = np.random.RandomState(random_seed)

        #self.raw_traj_nums = len(trajectories)
        self.raw_traj_nums = min(len(trajectories),self.traj_lim)
        self.last_terminal = False
        tot_steps = 0

        cur_trajs = 0

        print(len(trajectories[0]['observations']))
        print(len(trajectories[0]['observations'][0]))
        for trajectory in trajectories:
            #print(len(trajectory['observations']))
            cur_trajs+=1
            if cur_trajs>self.traj_lim:
                break
            tot_steps +=len(trajectory['observations'])
            self.last_terminal = False
            for i in range(len(trajectory['observations'])):
                self._observations.append(trajectory['observations'][i])
                if self._action_dim is None:
                    self._actions.append(trajectory['actions'][i])
                else:
                    new_action = np.zeros(self._action_dim)
                    new_action[trajectory['actions'][i]] = 1
                    self._actions.append(new_action)
                if self.last_terminal:
                    self._rewards.append(np.array([0.0]))
                else:
                    self._rewards.append(trajectory['rewards'][i])
                self._next_obs.append(trajectory['next_observations'][i])
                self._terminals.append(trajectory['terminals'][i])
                # self._gamma_remain.append(trajectory['gamma_remain'][i])
                # self._future_reward.append(trajectory['future_reward'][i])
                # self._last_observation.append(trajectory['last_observation'][i])
                self._cur_step.append(i)
                if trajectory['terminals'][i][0]:
                    self.last_terminal = True
                # Discount is always 1 (infinite-horizon setting).
                # discount.append(1.0)
        self._actions=np.array(self._actions)
        self._rewards=np.array(self._rewards)
        self._terminals=np.array(self._terminals)
        self._observations=np.array(self._observations)
        self._next_obs=np.array(self._next_obs)
        # self._gamma_remain = np.array(self._gamma_remain)
        # self._future_reward = np.array(self._future_reward)
        # self._last_observation = np.array(self._last_observation)
        self._cur_step = np.array(self._cur_step)
        self._buffer_size = len(self._actions)

        self._raw_actions = self._actions
        self._raw_observations = self._observations
        self._raw_next_obs = self._next_obs
        self._raw_cur_step = self._cur_step
        self._raw_buffer_size = self._buffer_size

        self.mean_steps = tot_steps/self.raw_traj_nums
        self.mean_rwd = self._get_info()

        if buffer_limit is not None:
            if self._buffer_size<=buffer_limit:
                pass
            else:
                rets = self._np_rand_state.randint(0, self._buffer_size, buffer_limit)
                self._actions=self._actions[rets]
                self._rewards=self._rewards[rets]
                self._terminals=self._terminals[rets]
                self._observations=self._observations[rets]
                self._next_obs=self._next_obs[rets]
                self._cur_step = self._cur_step[rets]
                self._buffer_size = len(self._actions)

        print(self._buffer_size)

    def _np_randint(self, *args, **kwargs):
        rets = self._np_rand_state.randint(*args, **kwargs)
        return rets
    

    def _np_choice(self, *args, **kwargs):
        rets = self._np_rand_state.choice(*args, **kwargs)
        return rets


    def random_batch(self, batch_size, keys=None):
        indices = self._np_randint(0, self._buffer_size, batch_size)
        return self._get_batch_using_indices(indices, keys=keys)

    def _get_raw_batch_using_indices(self, indices, keys=['observations','actions','next_observations','cur_step']):
        ret_dict = {}
        ret_dict['next_observations'] = self._raw_next_obs[indices]
        ret_dict['actions'] = self._raw_actions[indices]
        ret_dict['observations'] = self._raw_observations[indices]
        ret_dict['cur_step'] = self._raw_cur_step[indices]
        return ret_dict

    def first_batch(self, batch_size, keys=None):
        indices = np.arange(0,batch_size)
        return self._get_batch_using_indices(indices, keys=keys)

    def get_batch_byindice(self,indices, keys=None):
        return self._get_batch_using_indices(indices, keys=keys)

    def last_batch(self,batch_size, keys=None):
        indices = np.arange(self._buffer_size-batch_size,self._buffer_size)
        return self._get_batch_using_indices(indices, keys=keys)
    
    
    def _get_batch_using_indices(self, indices, keys=None):
        if keys is None:
            keys = set(
                ['observations', 'actions', 'rewards',
                'terminals', 'next_observations','cur_step'] #'ret','gae'
            )
        obs_to_return = self._observations[indices]
        next_obs_to_return = self._next_obs[indices]
        ret_dict = {}
        if 'observations' in keys: 
            ret_dict['observations'] = obs_to_return
        if 'actions' in keys: 
            ret_dict['actions'] = self._actions[indices]
        if 'rewards' in keys: 
            ret_dict['rewards'] = self._rewards[indices]
        if 'terminals' in keys: 
            ret_dict['terminals'] = self._terminals[indices]
        if 'next_observations' in keys: 
            ret_dict['next_observations'] = next_obs_to_return
        # if 'gamma_remain' in keys:
        #     ret_dict['gamma_remain'] = self._gamma_remain[indices]
        # if 'future_reward' in keys:
        #     ret_dict['future_reward'] = self._future_reward[indices]
        # if 'last_observation' in keys:
        #     ret_dict['last_observation'] = self._last_observation[indices]
        if 'cur_step' in keys:
            ret_dict['cur_step'] = self._cur_step[indices]
        # if 'gae' in keys:
        #     ret_dict['gae'] = self.gae[indices]
        # if 'ret' in keys:
        #     ret_dict['ret'] = self.ret[indices]
        return ret_dict
    
    def num_steps_can_sample(self):
        return self._buffer_size

    def _get_info(self,batch_size=256):
        tot_sum = 0.0
        last_index = 0
        data_size = self._buffer_size
        iter_nums = int(data_size/batch_size)
        keys = ['cur_step','rewards','observations','actions']
        for i in range(iter_nums):
            batch = self._get_batch_using_indices(np.arange(last_index,last_index+batch_size), keys)
            cur_step = np.squeeze(np.array(batch['cur_step']))
            rewards = np.squeeze(np.array(batch['rewards']))
            gamma_coef = np.power(self.gamma,cur_step)
            cur_sum = np.sum(rewards*gamma_coef)
            tot_sum+=cur_sum
            last_index+=batch_size
        batch = self._get_batch_using_indices(np.arange(last_index,data_size), keys)
        cur_step = np.squeeze(np.array(batch['cur_step']))
        rewards = np.squeeze(np.array(batch['rewards']))
        gamma_coef = np.power(self.gamma,cur_step)
        cur_sum = np.sum(rewards*gamma_coef)
        tot_sum+=cur_sum

        # print(self._buffer_size)
        # print(self.raw_traj_nums)
        mean_sum = tot_sum/self.raw_traj_nums
        # print(mean_sum)
        # exit(1)
        return mean_sum
