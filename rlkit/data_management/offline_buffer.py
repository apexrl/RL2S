from collections import defaultdict
import random as python_random
from random import sample
from itertools import starmap
from functools import partial

import numpy as np

from rlkit.data_management.replay_buffer import ReplayBuffer


class OfflineBuffer():
    '''
        THE MAX LENGTH OF AN EPISODE SHOULD BE STRICTLY SMALLER THAN THE max_replay_buffer_size
        OTHERWISE THERE IS A BUG IN TERMINATE_EPISODE

        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
    '''
    def __init__(self, trajectories, random_seed=1995):
        self._actions = []
        self._rewards = []
        self._terminals = []
        self._observations = []
        self._next_obs = []

        self._gamma_remain = []
        self._future_reward = []
        self._last_observation = []
        self._np_rand_state = np.random.RandomState(random_seed)

        print(len(trajectories[0]['observations']))
        print(len(trajectories[0]['observations'][0]))
        for trajectory in trajectories:
            for i in range(len(trajectory['observations'])):
                self._observations.append(trajectory['observations'][i])
                self._actions.append(trajectory['actions'][i])
                self._rewards.append(trajectory['rewards'][i])
                self._next_obs.append(trajectory['next_observations'][i])
                self._terminals.append(trajectory['terminals'][i])
                self._gamma_remain.append(trajectory['gamma_remain'][i])
                self._future_reward.append(trajectory['future_reward'][i])
                self._last_observation.append(
                    trajectory['last_observation'][i])
                # Discount is always 1 (infinite-horizon setting).
                # discount.append(1.0)
        self._actions = np.array(self._actions)
        self._rewards = np.array(self._rewards)
        self._terminals = np.array(self._terminals)
        self._observations = np.array(self._observations)
        self._next_obs = np.array(self._next_obs)
        self._gamma_remain = np.array(self._gamma_remain)
        self._future_reward = np.array(self._future_reward)
        self._last_observation = np.array(self._last_observation)
        self._buffer_size = len(self._actions)

    def _np_randint(self, *args, **kwargs):
        rets = self._np_rand_state.randint(*args, **kwargs)
        return rets

    def _np_choice(self, *args, **kwargs):
        rets = self._np_rand_state.choice(*args, **kwargs)
        return rets

    def random_batch(self, batch_size, keys=None):
        indices = self._np_randint(0, self._buffer_size, batch_size)
        return self._get_batch_using_indices(indices, keys=keys)

    def expert_random_batch(self, batch_size, keys=None, incept=0.5):
        low_index = int(incept * self._buffer_size)
        indices = self._np_randint(low_index, self._buffer_size, batch_size)
        return self._get_batch_using_indices(indices, keys=keys)

    def first_batch(self, batch_size, keys=None):
        indices = np.arange(0, batch_size)
        return self._get_batch_using_indices(indices, keys=keys)

    def get_batch_byindice(self, indices, keys=None):
        return self._get_batch_using_indices(indices, keys=keys)

    def last_batch(self, batch_size, keys=None):
        indices = np.arange(self._buffer_size - batch_size, self._buffer_size)
        return self._get_batch_using_indices(indices, keys=keys)

    def _get_batch_using_indices(self, indices, keys=None):
        if keys is None:
            keys = set([
                'observations', 'actions', 'rewards', 'terminals',
                'next_observations', 'gamma_remain', 'future_reward',
                'last_observation'
            ])
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
        if 'gamma_remain' in keys:
            ret_dict['gamma_remain'] = self._gamma_remain[indices]
        if 'future_reward' in keys:
            ret_dict['future_reward'] = self._future_reward[indices]
        if 'last_observation' in keys:
            ret_dict['last_observation'] = self._last_observation[indices]
        return ret_dict

    def num_steps_can_sample(self):
        return self._buffer_size


class SimpleOfflineBuffer(OfflineBuffer):
    def __init__(self, trajectories, random_seed=1995):
        self._actions = []
        self._rewards = []
        self._terminals = []
        self._observations = []
        self._next_obs = []
        self._np_rand_state = np.random.RandomState(random_seed)

        for trajectory in trajectories:
            for i in range(len(trajectory['observations'])):
                self._observations.append(trajectory['observations'][i])
                self._actions.append(trajectory['actions'][i])
                self._rewards.append(trajectory['rewards'][i])
                self._next_obs.append(trajectory['next_observations'][i])
                self._terminals.append(trajectory['terminals'][i])
        self._actions = np.array(self._actions)
        self._rewards = np.array(self._rewards)
        self._terminals = np.array(self._terminals)
        self._observations = np.array(self._observations)
        self._next_obs = np.array(self._next_obs)
        self._buffer_size = len(self._actions)

    def _get_batch_using_indices(self, indices, keys=None):
        if keys is None:
            keys = set([
                'observations', 'actions', 'rewards', 'terminals',
                'next_observations'])
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
        return ret_dict
