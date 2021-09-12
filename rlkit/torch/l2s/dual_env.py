import gym
import numpy as np
import torch
from .utils import *
# from rlkit.torch.l2s.utils import unnormalize_obs, normalize_acts,normalize_obs

def dual_space(p_state_space, p_action_space):
    # If action space is discrete
    n_action = None
    if isinstance(p_action_space, gym.spaces.Discrete):
        n_action = p_action_space.n
        p_action_space = gym.spaces.Box(np.zeros([n_action]), np.ones([n_action]))
    # Concatenate the state space and action space to form the new state space
    d_state_space = gym.spaces.Box(
        np.concatenate([p_state_space.low, p_action_space.low]),
        np.concatenate([p_state_space.high, p_action_space.high])
    )
    d_action_space = p_state_space
    return d_state_space, d_action_space

class DualEnv(gym.Env):
    def __init__(self, 
        primal_env, 
        model=None, 
        max_length=200,
        env_observation_mean = None,
        env_observation_std = None,
        env_action_mean = None,
        env_action_std = None,):
        """[summary]

        Args:
            primal_env ([type]): [description], always unnormalized
            model ([type], optional): [description]. pytorch module, i.e. policy, take unnormalizde data as input
        """
        if isinstance(primal_env, str):
            primal_env = gym.make(primal_env)
        self.primal_env = primal_env
        self.model = model
        self.max_length = max_length
        if isinstance(self.primal_env.action_space, gym.spaces.Discrete):
            self.action_n = self.primal_env.action_space.n
        else:
            self.action_n = 0
        self.observation_space, self.action_space = dual_space(self.primal_env.observation_space,
                                                               self.primal_env.action_space)
        self.last_done = False

        self.env_observation_mean = env_observation_mean
        self.env_observation_std = env_observation_std
        self.env_action_mean = env_action_mean
        self.env_action_std = env_action_std
        self.if_normalize_obs = False
        self.if_normalize_acts = False
        self.steps = 0
        if self.env_observation_mean is not None:
            self.if_normalize_obs = True
        if self.env_action_mean is not None:
            self.if_normalize_acts = True

    def reset(self):
        # assert self.model is not None
        state = self.primal_env.reset() #data from primal_env is always unnormalized 
        self.steps = 0
        raw_state = self.primal_env._get_obs(state)
        #raw_state = get_raw_ob_from_state(state)

        action,_ = self.model.get_action(raw_state,deterministic=True)
        # action, _ = self.model.predict(state)
        if self.action_n:
            action = np.eye(1, self.action_n, action)
        else:
            if self.if_normalize_acts:
                action = normalize_acts(action,self.env_action_mean,self.env_action_std)

        if self.if_normalize_obs:
            state = normalize_obs(state,self.env_observation_mean,self.env_observation_std)
        
        return np.concatenate((state, action.flatten()))

    def step(self,state):
        assert self.model is not None
        
        if self.if_normalize_obs:
            unnorm_state = unnormalize_obs(state,self.env_observation_mean,self.env_observation_std)
            raw_unnorm_state = self.primal_env._get_obs(unnorm_state)
            action,_ = self.model.get_action(raw_unnorm_state,deterministic=True)
        else:
            raw_state = self.primal_env._get_obs(state)
            action,_ = self.model.get_action(raw_state,deterministic=True)
        # print(state, self.is_done(state), action)
        if self.action_n:
            action = np.eye(1, self.action_n, action)
        else:
            if self.if_normalize_acts:
                action = normalize_acts(action,self.env_action_mean,self.env_action_std)
        
        self.steps += 1
        dual_state = np.concatenate((state, action.flatten()))

        done = self.is_done(state)

        if done:
            self.last_done=True
        return  dual_state, 0, done, {}

    def get_action(self, obs_np,deterministic=True):
        if self.if_normalize_obs:
            obs_np = unnormalize_obs(obs_np,self.env_observation_mean,self.env_observation_std)
        
        obs_np = self.primal_env._get_obs(obs_np)

        action,_ = self.model.get_action(obs_np,deterministic=True)
        if self.if_normalize_acts:
            action = normalize_acts(action,self.env_action_mean,self.env_action_std)
        return action,_


    def set_num_steps_total(self, t):
        pass

    def render(self, mode='human'):
        pass

    def is_done(self, state):
        if self.if_normalize_obs:
            state = unnormalize_obs(state,self.env_observation_mean,self.env_observation_std)
        return self.steps >= self.max_length or self.primal_env.is_done(state)

        