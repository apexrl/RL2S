import numpy as np
import gym
import heapq,os,joblib

def flatten_space(space, v):
    if isinstance(space, gym.spaces.Discrete):
        new_v = np.zeros(space.n)
        new_v[v] = 1
        return new_v
    elif isinstance(space, gym.spaces.Box):
        return v.flatten()
    else:
        raise RuntimeError("Not supported")

def dual_state(env, p_state, p_action):
    return np.concatenate([
        flatten_space(env.observation_space, p_state),
        flatten_space(env.action_space, p_action)
    ])

def get_non_absorbing_state(s):
    return np.concatenate([s, [1]])

def get_absorbing_state(s):
    return np.zeros(len(s)+1)

def normalize_obs(obs,obs_mean,obs_std):
    normal_obs = (obs - obs_mean) / (obs_std+1e-6)
    return normal_obs

def unnormalize_obs(obs,obs_mean,obs_std):
    unnormal_obs = obs*obs_std+obs_mean
    return unnormal_obs

def normalize_acts(acts,acts_mean,acts_std):
    noraml_acts = (acts-acts_mean)/(acts_std+1e-6)
    return noraml_acts

def unnormalize_acts(acts,acts_mean,acts_std):
    unnormal_acts = acts*acts_std+acts_mean
    return unnormal_acts

def get_raw_ob_from_state(state):
        """[used in true hopper env]
        The state is not batch_form
        Returns:
            [type]: [description]
        """
        if len(state.shape)>=2:
            raise ValueError("Our code do not support batch_norm env")
        per_size = state.shape[-1]
        qpos = state[0:per_size]
        qvel = state[per_size:]
        return np.concatenate([
            qpos.flat[1:],
            np.clip(qvel.flat, -10, 10)
        ])

def compute_stat(base_dir,id_list):
    cur_obs = None
    cur_acts = None
    cur_length = 0
    train_data = base_dir + "train_data/"
    for i in id_list:
        last_path = "dataset_policy_%d.pkl"%i
        data_path = train_data+last_path
        trajs = joblib.load(data_path)
        for traj in trajs:
            tmp_obs = np.array(traj["observations"])
            tmp_acts = np.array(traj["actions"])
            cur_length +=len(tmp_obs)
            if cur_obs is None:
                cur_obs = tmp_obs
                cur_acts = tmp_acts
            else:
                cur_obs = np.concatenate((cur_obs,tmp_obs),axis=0)
                cur_acts = np.concatenate((cur_acts,tmp_acts),axis=0)
    obs_mean = np.mean(cur_obs,axis=0)
    obs_std = np.std(cur_obs,axis=0)
    acts_mean = np.mean(cur_acts,axis=0)
    acts_std = np.std(cur_acts,axis=0)
    return obs_mean,obs_std,acts_mean,acts_std