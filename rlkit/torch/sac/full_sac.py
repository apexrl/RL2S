from collections import OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.trainer import Trainer
from rlkit.core.eval_util import create_stats_ordered_dict
import gtimer as gt
from rlkit.core import logger
# from rlkit.core.loss import LossFunction, LossStatistics
# from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.offline_utils import clip_gradient


class FullSoftActorCritic(Trainer):
    def __init__(
            self,
            policy,
            qf1,
            qf2,

            action_space,
            discount=0.99,
            reward_scale=1.0,
            
            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            beta_1=0.9,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            target_qf1=None,
            target_qf2=None,
    ):
        super().__init__()
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        if target_qf1 is None:
            self.target_qf1 = self.qf1.copy()
        else:
            self.target_qf1 = target_qf1
        if target_qf2 is None:
            self.target_qf2 = self.qf2.copy()
        else:
            self.target_qf2 = target_qf2
        
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy is None:
                # Use heuristic value from SAC paper
                self.target_entropy = -np.prod(
                    action_space.shape).item()
            else:
                self.target_entropy = target_entropy
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
            #betas=(beta_1, 0.999)
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

    def train_step(self, batch):
        policy_loss,qf1_loss,qf2_loss,alpha_loss,stats = self.compute_loss(batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        """
        Update networks
        """
        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self._n_train_steps_total += 1

        self.try_update_target_networks()
        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False

    def try_update_target_networks(self):
        if self._n_train_steps_total % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(
            self.qf1, self.target_qf1, self.soft_target_tau
        )
        ptu.soft_update_from_to(
            self.qf2, self.target_qf2, self.soft_target_tau
        )

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
    ):
        rewards = batch['rewards']
        if len(rewards.size())==1:
            rewards = torch.unsqueeze(rewards,dim=-1)
        terminals = batch['terminals']
        if len(terminals.size())==1:
            terminals = torch.unsqueeze(terminals,dim=-1)
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy and Alpha Loss
        """
        policy_outputs = self.policy(obs, return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # print('new_actions',new_actions.size())
        
        # #log_pi = log_pi.unsqueeze(-1)
        # print('log_pi',log_pi.size())
        # print('self.target_entropy',self.target_entropy)

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(obs, new_actions),
            self.qf2(obs, new_actions),
        )
        # print('alpha',alpha.size())
        # print('alpha_loss',alpha_loss.size())
        # print('q_new_actions',q_new_actions.size())
        policy_loss = (alpha*log_pi - q_new_actions).mean()
        # print('policy_loss',policy_loss.size())
        # print((alpha*log_pi).size())

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)

        next_policy_outputs = self.policy(next_obs, return_log_prob=True)
        new_next_actions, next_policy_mean, next_policy_log_std, new_log_pi = next_policy_outputs[:4]
        # new_log_pi = new_log_pi.unsqueeze(-1)
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values

        # print('target_q_values',target_q_values.size())
        # print('q_target',q_target.size())
        # print('q1_pred',q1_pred.size())
        # print('q2_pred',q2_pred.size())

        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            # policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
            # eval_statistics.update(policy_statistics)
            if self.use_automatic_entropy_tuning:
                eval_statistics['Alpha'] = alpha.item()
                eval_statistics['Alpha Loss'] = alpha_loss.item()

        # loss = SACLosses(
        #     policy_loss=policy_loss,
        #     qf1_loss=qf1_loss,
        #     qf2_loss=qf2_loss,
        #     alpha_loss=alpha_loss,
        # )

        return policy_loss,qf1_loss,qf2_loss,alpha_loss,eval_statistics

    def get_eval_statistics(self):
        # stats = super().get_diagnostics()
        # stats.update(self.eval_statistics)
        # return stats
        return self.eval_statistics
        

    def end_epoch(self):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    @property
    def optimizers(self):
        return [
            self.alpha_optimizer,
            self.qf1_optimizer,
            self.qf2_optimizer,
            self.policy_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )