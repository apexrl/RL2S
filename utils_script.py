import yaml
import argparse
import joblib
import numpy as np
import os,sys,inspect
import pickle as pkl
import joblib,pprint
import gym
import argparse
import pandas as pd
import torch,random


from gym.spaces import Dict
from rlkit.envs import get_env

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.core import logger
from rlkit.torch.core import np_to_pytorch_batch

from rlkit.torch.networks import FlattenMlp
from rlkit.data_management.offline_buffer import OfflineBuffer,SimpleOfflineBuffer
from rlkit.envs.wrappers import ScaledEnv
from rlkit.launchers import support_config as config
from rlkit.torch.offline_utils import compute_dataset_q
from rlkit.samplers import PathSampler,rollout
from rlkit.samplers.l2s_sampler import rolloutSimMujoco, L2SPathSampler
from rlkit.torch.sac.policies import MakeDeterministic,L2SGaussianPolicy
from rlkit.torch.l2s.utils import normalize_acts,normalize_obs
from rlkit.envs import get_env
from rlkit.torch.l2s.customized_env.hopper import HopperWrapper
from rlkit.torch.l2s.customized_env.walker import WalkerWrapper
from rlkit.torch.l2s.customized_env.halfcheetah import HalfcheetahWrapper
from rlkit.torch.l2s.customized_env.ant import AntWrapper
from rlkit.data_management.traj_l2s_buffer import TrajL2SBuffer
from rlkit.torch.l2s.learn_env_sac import LearnEnv
from rlkit.torch.l2s.dual_env import DualEnv
from rlkit.torch.l2s.learn_env_sac import LearnEnv, LearnEnvForTrain
from rlkit.core import logger, eval_util
import math,copy
from scipy import stats

seed = 9999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def offline_evaluation(config_file,model_index, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    wrapper_dict = {'Hopper':HopperWrapper,'Halfcheetah':HalfcheetahWrapper,'Walker':WalkerWrapper,'Ant':AntWrapper}
    #load config
    with open(config_file, 'r') as spec_file:
        spec_string = spec_file.read()
        variant = yaml.load(spec_string)

    if variant['num_gpu_per_worker'] > 0:
        print('\n\nUSING GPU\n\n')
        ptu.set_gpu_mode(True, 0)
    #load policy
    with open('l2s_demos_listing.yaml', 'r') as f:
        listings = yaml.load(f.read())
    
    expert_data_test_index = listings[variant['expert_name']]['test_data_index']
    expert_base_dir = listings[variant['expert_name']]['base_dir']
    test_policy_set = []

    for i in expert_data_test_index:
        policy_path = os.path.join(expert_base_dir,'test_policy/policy_%d.pkl'%i)
        dual_env = joblib.load(policy_path)
        test_policy_set.append(dual_env)
    print("num of test dual env: ", len(test_policy_set))
    print("data load done")
    #load dynamic
    env_specs = variant['env_specs']
    test_env = get_env(env_specs)  # test_env
    test_env.seed(env_specs['eval_env_seed'])
    test_env = wrapper_dict[variant["expert_name"]](test_env)
    dynamic_dir = listings[variant['expert_name']]['dynamic_dir']
    dynamic_dir = os.path.join(dynamic_dir, '%s.pkl' % model_index)
    dynamic_model = joblib.load(dynamic_dir)['exploration_policy']
    learned_test_env = LearnEnvForTrain(
        test_env,
        dynamic_model,
        **variant['learned_env_params'],
    )
    #to gpu
    if ptu.gpu_enabled():
        for i in test_policy_set:
            i.to(ptu.device)
        dynamic_model.to(ptu.device)
        #algorithm.to(ptu.device)
        
    #begin evaluation
    res = []
    for i in range(len(test_policy_set)):
        learned_eval_sampler = eval_sampler = PathSampler(
            learned_test_env,
            test_policy_set[i],
            variant["num_steps_per_eval"],
            variant["max_path_length"],
            no_terminal=variant["no_terminal"],
        )
        learned_test_paths = learned_eval_sampler.obtain_samples()
        average_returns = eval_util.get_average_returns(learned_test_paths)
        res.append(average_returns)
    print(res)

    return res

def normalize_data(raw_data):
    raw_max = max(raw_data)
    raw_min = min(raw_data)
    res = []
    for i in raw_data:
        res.append((i-raw_min)/(raw_max-raw_min))
    return res



if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=int, default=0,help='choose the util')
    parser.add_argument('-d', '--dataset', type=str, default="cartpole",help='choose the dataset')
    parser.add_argument('-g', '--gpu', help='gpu id', type=int, default=0)
    parser.add_argument('--id', type=int, default=0,help='choose the data of offline evaluation')
    args = parser.parse_args()
    target=args.target
    env_name = args.dataset
    env_dict = {'hopper':'hopper','halfcheetah':'halfcheetah','walker':'walker','ant':'ant'}

    if target==0:
        final_res = []
        config_file = "exp_specs/%s_ope.yaml"%(env_name)
        model_list =['sn_300_best_r','sn_300_best_ur']

        for i in model_list:
            print(i)
            res = offline_evaluation(config_file,i,args.gpu)
            final_res.append(res)
        with open("./l2s_dataset/ope/%s.pkl"%env_name,'wb') as f:
            pkl.dump(final_res,f)
    elif target==1:
        expertname_dict = {"hopper":"Hopper","walker":"Walker","halfcheetah":"Halfcheetah","ant":"Ant"}
        with open('l2s_demos_listing.yaml', 'r') as f:
            listings = yaml.load(f.read())
        expert_data_test_index = listings[expertname_dict[env_name]]['test_data_index']
        perform_file = './l2s_dataset/%s/progress.csv' % env_name # Load real performance
        data = pd.read_csv(perform_file)
        perform_array = []
        for i in expert_data_test_index:
            perform_array.append(data['AverageReturn'][i])
        print(perform_array)
        seed = 9999
        set_seed(seed)
        file_path = "./l2s_dataset/ope/%s.pkl" % (env_name)
        raw_data = pkl.load(open(file_path,'rb')) 
        
        index_array = copy.deepcopy(perform_array)
        index_array = stats.rankdata(index_array)
        index_len = len(index_array)
        print("---------------------------------------------compute AUC-----------------------------------")
        for eval_per in raw_data:
            cnt_p=0
            cnt_n=0
            for i in range(index_len):
                for j in range(index_len):
                    if j==i:
                        continue
                    if (index_array[i]-index_array[j])*(eval_per[i]-eval_per[j])>0:
                        cnt_p+=1
                    else:
                        cnt_n+=1
            auc = cnt_p / (index_len*(index_len-1))
            tau = (cnt_p - cnt_n) / (index_len*(index_len-1))
            print(auc,' ',tau)
        print("---------------------------------------------compute NDCG-----------------------------------")
        norm_perform_array = normalize_data(perform_array)
        # NDCG
        ratio = [1,2]
        for j in ratio:
            ndcg = []
            top_num = j
            for eval_per in raw_data:
                concat = [(eval_per[i],norm_perform_array[i]) for i in range(index_len)]
                concat.sort(key=lambda obj:obj[0], reverse=True) 
                new_array = copy.deepcopy(norm_perform_array)
                new_array.sort(reverse=True)
                dcg = 0.0
                max_dcg = 0.0
                for i in range(top_num):
                    tmp = (2**(concat[i][1]) - 1) / math.log(i+2,2)
                    dcg+=tmp
                    tmp = (2**(new_array[i])-1)/math.log(i+2,2)
                    max_dcg +=tmp
                ndcg.append(dcg/max_dcg)
            print(j,' ',top_num,' ',ndcg)

                


            

                
        


