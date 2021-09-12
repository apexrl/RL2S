import gym, time
import gtimer as gt
import numpy as np
from collections import OrderedDict
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.l2s.utils import normalize_acts, normalize_obs, unnormalize_obs, unnormalize_acts


class LearnEnv(gym.Env):
    """
    A base class for all learning to simulate environment
    i.e., learned Env
    """
    def __init__(self, 
        env, 
        model,
        use_delta=False, 
        use_gail=False,
        env_observation_mean = None,
        env_observation_std = None,
        env_action_mean = None,
        env_action_std = None,
    ):
        """[summary]

        Args:
            env ([type]): inner_env
            model ([pytorch nn]): a learned policy, (i.e. learned dynamics) 
            use_delta (bool, optional): [description]. Defaults to False.
            use_gail (bool, optional): [description]. Defaults to False.
        """
        super(LearnEnv, self).__init__()
        self.model = model # the learned dynamic
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.is_done = env.is_done
        self.get_reward = env.get_reward
        self.predict = env.predict
        self.inner_env = env
        self.use_delta = use_delta
        self.use_gail = use_gail
        self.env_observation_mean = env_observation_mean
        self.env_observation_std = env_observation_std
        self.env_action_mean = env_action_mean
        self.env_action_std = env_action_std
        self.if_normalize_obs = False
        self.if_normalize_acts = False
        if self.env_observation_mean is not None:
            self.if_normalize_obs = True
        if self.env_action_mean is not None:
            self.if_normalize_acts = True

        
    def render(self):
        if self.if_normalize_obs:
            self.inner_env.set_state(unnormalize_obs(self.state,self.env_observation_mean,self.env_observation_std))
        else:
            self.inner_env.set_state(self.state)
        # self.inner_env.set_state(self.state[:-1] if self.use_gail else self.state)
        # print(self.inner_env.state, self.state)
        self.inner_env.render()

    def vec_action(self, action):
        if isinstance(self.action_space, gym.spaces.Discrete):
            one_hot = np.zeros(self.action_space.n)
            one_hot[action] = 1
            return one_hot
        else:
            return action

    def step_pb(self, action, state=None):
        """
            for deterministic environment ,return 1 is enough
            
        """
        assert not self.model is None
        if state is None:
            state = self.state
        action = self.vec_action(action)
        ob_input = np.concatenate([state, action])
        with torch.no_grad():
            state = torch.from_numpy(state).to(ptu.device)
            action = torch.from_numpy(action).to(ptu.device)
            if len(state.size())<2:
                state = torch.unsqueeze(state,dim=-1)
            if len(state.size())<2:
                action = torch.unsqueeze(action,dim=-1)
            log_prob = self.model.get_log_prob(state,action)
            log_prob = torch.squeeze(log_prob)
            prob = torch.exp(log_prob)
            prob = prob.detach().cpu().numpy()
        return float(prob[0])

    def seed(self, s):
        self.inner_env.seed(s)

    def reset(self):
        self.state = self.inner_env.reset()
        if self.if_normalize_obs:
            self.state = normalize_obs(self.state,self.env_observation_mean,self.env_observation_std)
        return self.state

    def step(self, action, deterministic=True):
        assert not self.model is None
        assert not self.state is None

        # _state = self.inner_env.predict(self.state, action)
        # done = self.is_done(_state)

        vec_action = self.vec_action(action)
        # vec_action = action
        old_state = self.state
        if self.use_gail:
            ob_input = np.concatenate([self.state, vec_action, [1]]) #TODO: The meaning of [1]???
        else:
            ob_input = np.concatenate([self.state, vec_action])
        
        predict_result, _ = self.model.get_action(ob_input, deterministic=deterministic) #action,action_info, 0 or 1

        if self.use_delta:
            self.state = predict_result + self.state
        else:
            self.state = predict_result
        
        self.set_state(self.state)

        if self.if_normalize_obs:
            raw_state = unnormalize_obs(self.state,self.env_observation_mean,self.env_observation_std)
            raw_old_state = unnormalize_obs(old_state,self.env_observation_mean,self.env_observation_std)
        else:
            raw_state = self.state
            raw_old_state = old_state
        if self.if_normalize_acts:
            raw_action = unnormalize_acts(action, self.env_action_mean,self.env_action_std)
        else:
            raw_action = action

        reward = self.get_reward(raw_old_state, raw_action, raw_state) # take unnormalized state as input
        done = self.is_done(raw_state) # # take unnormalized state as input

        return self.state, reward, done, {}
        
    def set_state(self, state):
        self.state = state
        # if self.if_normalize_obs:
        #     self.inner_env.set_state(unnormalize_obs(state,self.env_observation_mean,self.env_observation_std))
        # else:
        #     self.inner_env.set_state(state)


class LearnEnvForTrain(gym.Env):
    """
    A base class for all learning to simulate environment
    i.e., learned Env
    """
    def __init__(self, 
        env, 
        model,
        use_obs = False,
        use_delta=False, 
        use_gail=False,
        env_observation_mean = None,
        env_observation_std = None,
        env_action_mean = None,
        env_action_std = None,
        rwd_type = 0,
        penalty_coef = 1.0,
        deterministic = True,
    ):
        """[summary]
        the input and the output of its public methods is unnormalized
        its hidden state is unnormalized as well
        Args:
            env ([type]): inner_env
            model ([pytorch nn]): a learned policy, (i.e. learned dynamics) 
            use_delta (bool, optional): [description]. Defaults to False.
            use_gail (bool, optional): [description]. Defaults to False.
            rwd_type (int, optional): 0: raw reward, 1: reward - var
        """
        super(LearnEnvForTrain, self).__init__()
        self.model = model # the learned dynamic
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.is_done = env.is_done
        self.get_reward = env.get_reward
        self.predict = env.predict
        self.inner_env = env
        self.use_obs = use_obs
        self.use_delta = use_delta
        self.use_gail = use_gail
        self.env_observation_mean = env_observation_mean
        self.env_observation_std = env_observation_std
        self.env_action_mean = env_action_mean
        self.env_action_std = env_action_std
        self.if_normalize_obs = False
        self.if_normalize_acts = False
        if self.env_observation_mean is not None:
            self.if_normalize_obs = True
            self.env_observation_mean = np.array(self.env_observation_mean)
            self.env_observation_std = np.array(self.env_observation_std)
        if self.env_action_mean is not None:
            self.if_normalize_acts = True
            self.env_action_mean = np.array(self.env_action_mean)
            self.env_action_std = np.array(self.env_action_std)
        self.rwd_type = rwd_type
        self.penalty_coef = penalty_coef
        self.deterministic = deterministic

    def render(self):
        if self.if_normalize_obs:
            self.inner_env.set_state(unnormalize_obs(self.state,self.env_observation_mean,self.env_observation_std))
        else:
            self.inner_env.set_state(self.state)
        # self.inner_env.set_state(self.state[:-1] if self.use_gail else self.state)
        # print(self.inner_env.state, self.state)
        self.inner_env.render()

    def vec_action(self, action):
        if isinstance(self.action_space, gym.spaces.Discrete):
            one_hot = np.zeros(self.action_space.n)
            one_hot[action] = 1
            return one_hot
        else:
            return action

    def step_pb(self, action, state=None):
        """
            depreacated
            for deterministic environment ,return 1 is enough
            
        """
        pass

    def seed(self, s):
        self.inner_env.seed(s)

    def reset(self):
        self.state = self.inner_env.reset() # unnormalized
        if self.use_obs:
            obs = self.inner_env._get_obs(self.state)
        else:
            obs = self.state
        return obs
    
    def get_obs(self,state):
        return self.inner_env._get_obs(state)
    
    def reset_raw(self):
        self.state = self.inner_env.reset() # unnormalized
        return self.state

    def step(self, action, deterministic=True):
        """[summary]

        Args:
            action ([type]): unnormalized
            deterministic (bool, optional): [description]. Defaults to True. deprecated
        """
        assert not self.model is None
        assert not self.state is None

        # _state = self.inner_env.predict(self.state, action)
        # done = self.is_done(_state)

        vec_action = self.vec_action(action)
        old_state = self.state
        if self.if_normalize_obs:
            norm_old_state = normalize_obs(old_state,self.env_observation_mean,self.env_observation_std)
            norm_cur_state = normalize_obs(self.state,self.env_observation_mean,self.env_observation_std)
        else:
            norm_old_state = old_state
            norm_cur_state = self.state
        
        if self.if_normalize_acts:
            norm_vec_action = normalize_acts(vec_action, self.env_action_mean,self.env_action_std)
        else:
            norm_vec_action = vec_action

        if self.use_gail:
            ob_input = np.concatenate([norm_cur_state, norm_vec_action, [1]]) #TODO: The meaning of [1]???
        else:
            ob_input = np.concatenate([norm_cur_state, norm_vec_action])
        
        predict_result, _ = self.model.get_action(ob_input, deterministic=self.deterministic) #action,action_info, 0 or 1, normalized

        if self.use_delta:
            if self.if_normalize_obs:
                new_state = norm_cur_state+predict_result
                new_state = unnormalize_obs(new_state,self.env_observation_mean,self.env_observation_std)
            else:
                new_state = norm_cur_state+predict_result
        else:
            if self.if_normalize_obs:
                new_state = unnormalize_obs(predict_result,self.env_observation_mean,self.env_observation_std)
            else:
                new_state = predict_result
        
        self.state = new_state
        # self.set_state(self.state)

        reward = self.get_reward(old_state, vec_action, self.state) # take unnormalized state as input
        if self.rwd_type==1:
            std = self.model.get_std(ob_input, deterministic=self.deterministic)
            penalty = self.penalty_coef * np.sum(np.square(std))
            reward = reward - penalty

        done = self.is_done(self.state) # # take unnormalized state as input

        if self.use_obs:
            obs = self.inner_env._get_obs(self.state)
        else:
            obs = self.state
        

        return obs, reward, done, {}
        
    def set_state(self, state):
        """
        only used in train, instead of test
        """
        self.state = state
        if self.use_obs:
            state = self.inner_env._get_obs(state)
        return state
    
    def get_step(self,state,action):
        assert not self.model is None

        # _state = self.inner_env.predict(self.state, action)
        # done = self.is_done(_state)

        vec_action = self.vec_action(action)
        if self.if_normalize_obs:
            norm_cur_state = normalize_obs(state,self.env_observation_mean,self.env_observation_std)
        else:
            norm_cur_state = state
        
        if self.if_normalize_acts:
            norm_vec_action = normalize_acts(vec_action, self.env_action_mean,self.env_action_std)
        else:
            norm_vec_action = vec_action

        if self.use_gail:
            ob_input = np.concatenate([norm_cur_state, norm_vec_action, [1]]) #TODO: The meaning of [1]???
        else:
            ob_input = np.concatenate([norm_cur_state, norm_vec_action])
        
        predict_result, _ = self.model.get_action(ob_input, deterministic=self.deterministic) #action,action_info, 0 or 1, normalized

        if self.use_delta:
            if self.if_normalize_obs:
                new_state = norm_cur_state+predict_result
                new_state = unnormalize_obs(new_state,self.env_observation_mean,self.env_observation_std)
            else:
                new_state = norm_cur_state+predict_result
        else:
            if self.if_normalize_obs:
                new_state = unnormalize_obs(predict_result,self.env_observation_mean,self.env_observation_std)
            else:
                new_state = predict_result
        
        reward = self.get_reward(state, vec_action, new_state) # take unnormalized state as input
        if self.rwd_type==1:
            std = self.model.get_std(ob_input, deterministic=self.deterministic)
            penalty = self.penalty_coef * np.sum(np.square(std))
            reward = reward - penalty

        done = self.is_done(new_state) # # take unnormalized state as input

        return new_state, reward, done, {}