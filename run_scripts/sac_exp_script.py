import yaml
import argparse
import joblib
import numpy as np
import os,sys,inspect
import pickle,gym

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir) 
print(sys.path)

from gym.spaces import Dict
from rlkit.envs import get_env

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.core import logger

from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.sac.full_sac import FullSoftActorCritic
from rlkit.torch.sac.sac_alg import SACAlgorithm
import pickle as pkl
from rlkit.envs.wrappers import ScaledEnv
from rlkit.launchers import sac_config as config
import torch
from rlkit.torch.l2s.customized_env.pendulum import PendulumWrapper
#from rlkit.torch.l2s.customized_env.hopper import HopperWrapperSim


def experiment(variant):
    env_specs = variant['env_specs']
    if env_specs['env_name'] in ['hopper','walker','halfcheetah','ant','humanoid']:
        env = get_env(env_specs)
        training_env = get_env(env_specs)
    else:
        wrapper_dict ={'Pendulum-v0':PendulumWrapper}
        env = wrapper_dict[env_specs['env_name']](gym.make(env_specs['env_name'])) 
        training_env = wrapper_dict[env_specs['env_name']](gym.make(env_specs['env_name'])) 
        # env = gym.make(env_specs['env_name'])
        # training_env = gym.make(env_specs['env_name'])

    env.seed(env_specs['eval_env_seed'])
    training_env.seed(env_specs['training_env_seed'])
    # if env_specs["env_name"]=='hopper':
    #     env = HopperWrapperSim(env)
    #     training_env = HopperWrapperSim(env)
    
    print('\n\nEnv: {}'.format(env_specs['env_name']))
    print('kwargs: {}'.format(env_specs['env_kwargs']))
    print('Obs Space: {}'.format(env.observation_space))
    print('Act Space: {}\n\n'.format(env.action_space))

    obs_space = env.observation_space
    act_space = env.action_space
    assert not isinstance(obs_space, Dict)
    assert len(obs_space.shape) == 1
    assert len(act_space.shape) == 1
    
    obs_dim = obs_space.shape[0]
    action_dim = act_space.shape[0]

    # build the policy models
    net_size = variant['policy_net_size']
    num_hidden = variant['policy_num_hidden_layers']
    qf1 = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    policy = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=num_hidden * [net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    # set up the adversaial algorithm
    trainer = FullSoftActorCritic(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        action_space=act_space,
        **variant['sac_params']
    )

    algorithm = SACAlgorithm(
        trainer,
        env=env,
        training_env=training_env,
        exploration_policy=policy,
        **variant['alg_params']
    )
    if ptu.gpu_enabled():
        algorithm.to(ptu.device)
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