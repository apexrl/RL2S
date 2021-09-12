import yaml
import argparse
import joblib
import numpy as np
import os,sys,inspect
import pickle

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
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
from rlkit.torch.l2s.downstream_sac_alg import DownstreamSACAlgorithm
import pickle as pkl
from rlkit.envs.wrappers import ScaledEnv
from rlkit.launchers import l2s_config as config
import torch
from rlkit.torch.l2s.customized_env.hopper import HopperWrapper
from rlkit.torch.l2s.customized_env.halfcheetah import HalfcheetahWrapper
from rlkit.torch.l2s.customized_env.walker import WalkerWrapper
from rlkit.torch.l2s.customized_env.ant import AntWrapper
from rlkit.torch.l2s.learn_env_sac import LearnEnv, LearnEnvForTrain

def experiment(variant):
    wrapper_dict = {'hopper':HopperWrapper,'halfcheetah':HalfcheetahWrapper,'walker':WalkerWrapper,'ant':AntWrapper}
    env_specs = variant['env_specs']
    env = get_env(env_specs) # test_env
    env.seed(env_specs['eval_env_seed'])
    training_env = get_env(env_specs)
    training_env.seed(env_specs['training_env_seed'])
    if wrapper_dict.get(env_specs["env_name"]) is not None:
        training_env = wrapper_dict[env_specs["env_name"]](training_env)
    
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

    # build training env
    with open('l2s_demos_listing.yaml', 'r') as f:
        listings = yaml.load(f.read())
    dynamic_dir = listings[variant['Env_name']]['dynamic_dir']
    dynamic_dir = os.path.join(dynamic_dir,'%s.pkl'%variant['dynamic_dir'])
    dynamic_model = joblib.load(dynamic_dir)['exploration_policy']
    learned_env = LearnEnvForTrain(
        training_env,
        dynamic_model,
        **variant['learned_env_params'],
    )
    # build the policy models
    net_size = variant['policy_net_size']
    num_hidden = variant['policy_num_hidden_layers']

    print("finetune a policy use learned model")
    q_dir = os.path.join(listings[variant['Env_name']]['q_dir'], "q_%d.pkl"%variant['policy_dir'])
    policy_dir = os.path.join(listings[variant['Env_name']]['policy_dir'], "policy_%d.pkl"%variant['policy_dir'])
    q_data =  joblib.load(q_dir)
    qf1 = q_data['qf1']
    qf2 = q_data['qf2']
    target_qf1 = q_data['target_qf1']
    target_qf2 = q_data['target_qf2']
    policy = joblib.load(policy_dir)

    # set up the adversaial algorithm
    trainer = FullSoftActorCritic(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        action_space=act_space,
        target_qf1 = target_qf1,
        target_qf2 = target_qf2,
        **variant['sac_params']
    )

    #Load buffer
    buffer_dir = listings[variant['Env_name']]['env_buffer_dir']
    env_state_buffer = joblib.load(buffer_dir)
    env_state_buffer = np.array(env_state_buffer)
    print("Load env state buffer done")

    algorithm = DownstreamSACAlgorithm(
        trainer,
        env_state_buffer=env_state_buffer,
        env=env,
        training_env=learned_env,
        exploration_policy=policy,
        **variant['alg_params']
    )
    if ptu.gpu_enabled():
        algorithm.to(ptu.device)
        dynamic_model.to(ptu.device)
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
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs,base_log_dir=config.L2S_LOCAL_LOG_DIR)
    experiment(exp_specs)