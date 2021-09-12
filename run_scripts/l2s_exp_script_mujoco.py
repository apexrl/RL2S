import yaml
import argparse,os
import joblib
import numpy as np
import os,sys,inspect,time
import pickle
import gym

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
print(sys.path)

from gym.spaces import Dict
from rlkit.envs import get_env

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed

from rlkit.torch.l2s.policies import L2SGaussianPolicy
from rlkit.torch.l2s.disc_models.simple_disc_models import MLPDisc
from rlkit.data_management.traj_l2s_buffer import TrajL2SBuffer
from rlkit.torch.l2s.contextual_bandit import ContextualBandit
from rlkit.torch.l2s.L2SGail import L2SGail
from rlkit.torch.l2s.customized_env.hopper import HopperWrapper
from rlkit.torch.l2s.customized_env.halfcheetah import HalfcheetahWrapper
from rlkit.torch.l2s.customized_env.walker import WalkerWrapper
from rlkit.torch.l2s.customized_env.ant import AntWrapper

from rlkit.launchers import l2s_config as config
from torch.nn import functional as F

wrapper_dict = {
    "hopper":HopperWrapper,
    "halfcheetah":HalfcheetahWrapper,
    "walker":WalkerWrapper,
    "ant":AntWrapper,
}

def experiment(variant):
    #LoEnv
    env_specs = variant['env_specs']
    env = get_env(env_specs)
    env = wrapper_dict[env_specs["env_name"]](env)
    env.seed(env_specs['eval_env_seed'])
    print('\n\nEnv: {}'.format(env_specs['env_name']))
    print('kwargs: {}'.format(env_specs['env_kwargs']))
    print('Obs Space: {}'.format(env.observation_space))
    print('Act Space: {}\n\n'.format(env.action_space))
    obs_space = env.observation_space
    act_space = env.action_space
    assert not isinstance(obs_space, Dict)
    assert len(obs_space.shape) == 1
    if isinstance(act_space,gym.spaces.Discrete):
        discrete = True
        buffer_action_dim = act_space.n
        action_dim = act_space.n
    else:
        discrete = False
        assert len(act_space.shape) == 1
        action_dim = act_space.shape[0]
        buffer_action_dim = None
    # Dim of dual Env
    obs_dim = obs_space.shape[0]
    dual_obs_dim = obs_dim + action_dim
    dual_action_dim = obs_dim

    #Load data
    with open('l2s_demos_listing.yaml', 'r') as f:
        listings = yaml.load(f.read())
    expert_data_train_index = listings[variant['expert_name']]['train_data_index']
    expert_data_test_index = listings[variant['expert_name']]['test_data_index']
    expert_base_dir = listings[variant['expert_name']]['base_dir']
    expert_dataset_list_train = []
    expert_dataset_list_test = []
    dual_env_set_train = []
    dual_env_set_test = []

    train_policy_steps_mean = []
    train_policy_rwd_mean = []
    test_policy_steps_mean = []
    test_policy_rwd_mean = []

    for i in expert_data_train_index:
        policy_path = os.path.join(expert_base_dir,'train_policy/policy_%d.pkl'%i)
        data_path = os.path.join(expert_base_dir,'train_data/dataset_policy_%d.pkl'%i)
        buffer_save_dict = joblib.load(data_path)
        buffer = TrajL2SBuffer(buffer_save_dict,gamma=variant['l2s_params']['gamma'],action_dim =buffer_action_dim,
                                buffer_limit=variant['buffer_limit'],traj_lim=10)
        train_policy_steps_mean.append(buffer.mean_steps)
        train_policy_rwd_mean.append(buffer.mean_rwd)
        expert_dataset_list_train.append(buffer)
        dual_env = joblib.load(policy_path)
        dual_env_set_train.append(dual_env)
    for i in expert_data_test_index:
        policy_path = os.path.join(expert_base_dir,'test_policy/policy_%d.pkl'%i)
        data_path = os.path.join(expert_base_dir,'test_data/dataset_policy_%d.pkl'%i)
        buffer_save_dict = joblib.load(data_path)
        buffer = TrajL2SBuffer(buffer_save_dict,gamma=variant['l2s_params']['gamma'],action_dim =buffer_action_dim,
                                buffer_limit=variant['buffer_limit'],traj_lim=10)
        test_policy_steps_mean.append(buffer.mean_steps)
        test_policy_rwd_mean.append(buffer.mean_rwd)
        expert_dataset_list_test.append(buffer)
        dual_env = joblib.load(policy_path)
        dual_env_set_test.append(dual_env)
    assert len(expert_dataset_list_train) == len(dual_env_set_train),"length should be same"
    assert len(expert_dataset_list_test) == len(dual_env_set_test),"length should be same"
    print("num of train dual env: ", len(expert_dataset_list_train))
    print("num of test dual env: ", len(expert_dataset_list_test))
    print(buffer_save_dict[0].keys())
    print("data load done")
    
    # build the discriminator model
    disc_model = MLPDisc(
        dual_obs_dim + dual_action_dim,
        num_layer_blocks=variant['disc_num_blocks'],
        hid_dim=variant['disc_hid_dim'],
        hid_act=variant['disc_hid_act'],
        if_clamp=variant['disc_if_clamp'],
        clamp_magnitude=variant['disc_clamp_magnitude']
    )

    # build the policy models
    net_size = variant['policy_net_size']
    num_hidden = variant['policy_num_hidden_layers']
    policy = L2SGaussianPolicy(   
        hidden_sizes=num_hidden * [net_size],
        obs_dim=dual_obs_dim,
        action_dim=dual_action_dim,
        hidden_activation = F.tanh,
    )

    trainer = ContextualBandit(
        policy=policy,
        action_space=act_space,
        discriminator=disc_model,
        **variant['bandit_params']
    )
    
    algorithm = L2SGail(
        model=trainer.policy,
        policy_trainer = trainer,
        discriminator= disc_model,
        dual_action_dim = dual_action_dim,
        dual_obs_dim = dual_obs_dim,
        primal_action_dim = action_dim,
        primal_obs_dim = obs_dim,
        env = env,
        dual_env_set_train= dual_env_set_train,
        dual_env_set_test=dual_env_set_test, 
        expert_dataset_list_train=expert_dataset_list_train,
        expert_dataset_list_test=expert_dataset_list_test,
        expert_traj_nums_train=10,
        expert_traj_nums_test=10,
        discrete=discrete,
        exploration_policy=policy,
        **variant['l2s_params']
    )

    print(ptu.gpu_enabled())
    algorithm.set_dataset_info(train_policy_steps_mean,train_policy_rwd_mean,test_policy_steps_mean,test_policy_rwd_mean)
    if ptu.gpu_enabled():
        algorithm.to(ptu.device)
    algorithm._init_index()
    algorithm.train()
    return 1


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    parser.add_argument('-g', '--gpu', help='gpu id', type=int, default=0)
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)

    if exp_specs['num_gpu_per_worker'] > 0:
        print('\n\nUSING GPU\n\n')
        ptu.set_gpu_mode(True, args.gpu)
    exp_id = exp_specs['exp_id']
    exp_prefix = exp_specs['exp_name']
    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs,base_log_dir=config.LOCAL_LOG_DIR)
    experiment(exp_specs)