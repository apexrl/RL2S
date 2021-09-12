from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
import torch.optim as optim

import rlkit.torch.pytorch_util as ptu
from rlkit.core.trainer import Trainer
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core import logger, eval_util



class SoftActorCriticNoV(Trainer):
    """
    version that:
        - uses reparameterization trick
        - no V function
        - multi Q function
    TODO: Recently in rlkit there is a version which only uses two Q functions
    as well as an implementation of entropy tuning but I have not implemented
    those
    """
    def __init__(
            self,
            policy,
            qfs,

            q_nums=2,
            q_loss_type = 0,
            minmax_weight = 0.75,
            reward_scale=1.0,
            discount=0.99,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            soft_target_tau=1e-2,

            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,

            optimizer_class=optim.Adam,
            beta_1=0.9,
    ):
        """[summary]
        Args:
            policy ([type]): [description]
            qfs ([type]): [description]
            q_loss_type (int, optional): [should in [0,1]]. 0:Q_target=r+minQ(s,a), 1:Q_target = r+lambda minQ(s,a)+(1-lambda) maxQ(s,a). Defaults to 0.
            minmax_weight (float, optional): [description]. Defaults to 0.75.
            reward_scale (float, optional): [description]. Defaults to 1.0.
            discount (float, optional): [description]. Defaults to 0.99.
            policy_lr ([type], optional): [description]. Defaults to 1e-3.
            qf_lr ([type], optional): [description]. Defaults to 1e-3.
            vf_lr ([type], optional): [description]. Defaults to 1e-3.
            soft_target_tau ([type], optional): [description]. Defaults to 1e-2.
            policy_mean_reg_weight ([type], optional): [description]. Defaults to 1e-3.
            policy_std_reg_weight ([type], optional): [description]. Defaults to 1e-3.
            optimizer_class ([type], optional): [description]. Defaults to optim.Adam.
            beta_1 (float, optional): [description]. Defaults to 0.9.

        Raises:
            ValueError: [description]
        """
        self.policy = policy
        self.qfs = qfs
        self.q_nums = q_nums
        self.qf_targets = []
        # for q_net in self.qfs:
        #     self.qf_targets.append(q_net.copy())
        self.qf_targets = self.qfs.copy()
        self.q_loss_type = q_loss_type
        self.minmax_weight = minmax_weight
        if self.q_loss_type==0:
            print("We use r+minQ(s,a) as the target")
        elif self.q_loss_type==1:
            print("We use r+(lambda)*minQ(s,a)+(1-lambda)*maxQ(s,a) as the target")
        else:
            raise ValueError("invalid q_loss type, shoule in [0,1]")
        self.reward_scale = reward_scale
        self.discount = discount
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight

        self.eval_statistics = None

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
            betas=(beta_1, 0.999)
        )

        # self.qfs_optimizer = []
        # for q_net in self.qfs:
        #     self.qfs_optimizer.append(optimizer_class(
        #         q_net.parameters(),
        #         lr=qf_lr, 
        #         betas=(beta_1,0.999))
        #     )
        self.qfs_optimizer = optimizer_class(
                self.qfs.parameters(),
                lr=qf_lr, 
                betas=(beta_1,0.999)
        )

    def train_step(self, batch):
        rewards = self.reward_scale * batch['rewards'] # [256,1]
        terminals = batch['terminals'] # [256]
        if len(terminals.size())==1:
            terminals = torch.unsqueeze(terminals,dim=-1) #[256,1]
        obs = batch['observations'] 
        actions = batch['actions']
        next_obs = batch['next_observations']
        # print("obs",obs.size())
        # print('actions',actions.size())
        # print('next_obs',next_obs.size())
        # print('terminals',terminals.size())
        # print('rewards',rewards.size())

        # q_preds = [q_net(obs,actions) for q_net in self.qfs]
        q_preds = self.qfs(obs,actions) #[256,20]
        # Make sure policy accounts for squashing functions like tanh correctly!
        """
        QF Loss
        """
        with torch.no_grad():
            #compute Q target
            policy_outputs = self.policy(next_obs, return_log_prob=True)
            next_actions = policy_outputs[0].detach()
            # q_preds_target = [q_net(next_obs,next_actions).detach() for q_net in self.qf_targets]
            # # print(q_preds_target[0].size()) #[256,1]
            # q_preds_target = torch.cat(q_preds_target,dim=-1) # [256,20]
            q_preds_target = self.qf_targets(next_obs,next_actions) # [256,20]
            min_next_Q = torch.min(q_preds_target,dim=-1).values
            min_next_Q = torch.unsqueeze(min_next_Q,-1)
            next_value = min_next_Q # [256,1]
            # print(next_value.size())
            # print(rewards.size())
            # print(terminals.size())
            if self.q_loss_type==0:       
                Q_target = rewards + (1. - terminals) * self.discount * next_value #[256,1]
            elif self.q_loss_type==1:
                max_next_Q = torch.max(q_preds_target,dim=-1).values
                max_next_Q = torch.unsqueeze(max_next_Q,-1)
                next_value = self.minmax_weight*next_value + (1.0-self.minmax_weight)*max_next_Q
                Q_target = rewards + (1. - terminals) * self.discount * next_value
            Q_target = Q_target.repeat(1,self.q_nums) # [256,20]
        
        # print(Q_target.size())
        # print(q_preds.size())
        # exit(1)
        # q_losses = [0.5*torch.mean((q_pred - Q_target.detach())**2) for q_pred in q_preds]
        q_loss = 0.5*torch.mean((q_preds - Q_target.detach())**2,dim=-1)
        q_loss = torch.sum(q_loss)
        # print(q_loss.device)
        # exit(1)

        # target_v_values = self.target_vf(next_obs)
        # q_target = rewards + (1. - terminals) * self.discount * target_v_values
        # qf1_loss = 0.5 * torch.mean((q1_pred - q_target.detach())**2)
        # qf2_loss = 0.5 * torch.mean((q2_pred - q_target.detach())**2)
        """
        Update Q networks
        """
        # for i in range(len(self.qfs)):
        #     self.qfs_optimizer[i].zero_grad()
        #     q_losses[i].backward()
        #     self.qfs_optimizer[i].step()
        self.qfs_optimizer.zero_grad()
        q_loss.backward()
        self.qfs_optimizer.step()

        """
        Policy Loss
        """
        policy_outputs = self.policy(obs, return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        # print("new_actions",new_actions.size()) [256,3]
        # q_preds_new = [q_net(obs,new_actions) for q_net in self.qfs]
        # q_preds_new_min = torch.cat(q_preds_new,dim=-1)
        q_preds_new = self.qfs(obs,new_actions)
        q_preds_new_min = torch.min(q_preds_new,dim=-1).values
        q_preds_new_min = torch.unsqueeze(q_preds_new_min,-1)

        policy_loss = torch.mean(- q_preds_new_min)
        # policy_loss = torch.mean(log_pi - min_next_Q)
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        policy_reg_loss = mean_reg_loss + std_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        """
        Update policy networks
        """

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self._update_target_network()

        """
        Save some statistics for eval
        """
        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            self.eval_statistics['Reward Scale'] = self.reward_scale
            # for i in range(len(self.qfs)):
            #     self.eval_statistics['QF%d Loss'%(i+1)] = np.mean(ptu.get_numpy(q_losses[i]))
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(q_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            # for i in range(len(self.qfs)):
            #     self.eval_statistics.update(create_stats_ordered_dict(
            #         'Q%d Predictions'%(i+1),
            #         ptu.get_numpy(q_preds[i]),
            #     ))
            self.eval_statistics.update(
                create_stats_ordered_dict('Q Predictions',
                ptu.get_numpy(q_preds))
            )
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

            # print(self.eval_statistics)
            # exit(1)


    @property
    def networks(self):
        res = [self.policy,self.qfs,self.qf_targets]

        # for i in self.qfs:
        #     res.append(i)
        # for i in self.qf_targets:
        #     res.append(i)
        
        # return [
        #     self.policy,
        #     self.qf1,
        #     self.qf2,
        #     self.vf,
        #     self.target_vf,
        # ]
        return res


    def _update_target_network(self):
        # for i in range(len(self.qfs)):
        #     ptu.soft_update_from_to(self.qfs[i], self.qf_targets[i], self.soft_target_tau)
        ptu.soft_update_from_to(self.qfs, self.qf_targets, self.soft_target_tau)


    def get_snapshot(self):
        # res = dict(
        #     policy=self.policy,
        # )
        # for i in range(len(self.qfs)):
        #     res["qf%d"%(i+1)]=self.qfs[i]
        #     res["target_qf%d"%(i+1)]=self.qf_targets[i]
        return dict(
            qfs=self.qfs,
            policy=self.policy,
            target_qf=self.qf_targets,
        )    

    def get_eval_statistics(self):
        return self.eval_statistics
    

    def end_epoch(self):
        self.eval_statistics = None

    def sample(self, obs,return_prob=True):
        '''
        specific for sac
        '''
        obs = torch.from_numpy(obs).float().to(ptu.device)
        self.policy.eval()
        action, mean, log_std, log_prob, expected_log_prob, std, mean_action_log_prob, pre_tanh_value = self.policy(obs, return_log_prob=return_prob)
        action = action.cpu().detach().numpy()
        if return_prob:
            prob = np.exp(log_prob.cpu().detach().numpy())
            prob = np.squeeze(prob)
            self.policy.train()
            return action, prob 
        else:
            self.policy.train()
            return action

    def sample_torch(self, obs):
        '''
        specific for sac
        '''
        with torch.no_grad():
            action, mean, log_std, log_prob, expected_log_prob, std, mean_action_log_prob, pre_tanh_value = self.policy(obs, return_log_prob=False)
            return action

    def get_min_Q(self,obs,actions):
        # q_preds_target = [q_net(obs,actions).detach() for q_net in self.qf_targets]
        # q_preds_target = torch.cat(q_preds_target,dim=-1)
        q_preds_target = self.qf_targets(obs,actions)
        min_next_Q = torch.min(q_preds_target,dim=-1).values
        min_next_Q = torch.unsqueeze(min_next_Q,-1)
        return min_next_Q
    
    def get_first_Q(self,obs,actions):
        return self.qfs.get_first(obs,actions)
        #return self.qfs.q_nets[0](obs,actions)
        #return self.qfs[0](obs,actions)

    def to(self, device):
        for net in self.networks:
            net.to(device)

    def train(self, start_epoch=0):
        self.pretrain()
        if start_epoch == 0:
            params = self.get_epoch_snapshot(-1)
            logger.save_itr_params(-1, params)
        self.training_mode(False)
        self._n_env_steps_total = start_epoch * self.num_env_steps_per_epoch
        gt.reset()
        gt.set_def_unique(False)
        self.start_training(start_epoch=start_epoch)

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)