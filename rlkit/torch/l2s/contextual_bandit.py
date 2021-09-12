from collections import OrderedDict, namedtuple
import numpy as np
import torch
import torch.optim as optim
import rlkit.torch.pytorch_util as ptu
from rlkit.core.trainer import Trainer
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.offline_utils import clip_gradient


class ContextualBandit(Trainer):
    def __init__(
            self,
            policy,
            action_space,

            discriminator = None,
            reward_scale=1.0,
            policy_lr=1e-3,
            optimizer_class=optim.Adam,

            beta_1=0.9,

            plotter=None,
            render_eval_paths=False,

            target_entropy=None,
    ):
        super().__init__()
        self.policy = policy
        self.discriminator =discriminator

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )

        self.reward_scale = reward_scale
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

    def train_step(self, batch):
        policy_loss,alpha_loss,stats = self.compute_loss(batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        """
        Update networks
        """

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        clip_gradient(self.policy_optimizer)
        self.policy_optimizer.step()

        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False

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


        _ , q_new_actions = self.discriminator(torch.cat([obs,new_actions],dim=-1))
        policy_loss = (-q_new_actions).mean()

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
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

        return policy_loss,0.0,eval_statistics

    def get_eval_statistics(self):
        return self.eval_statistics
        

    def end_epoch(self):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
        ]

    @property
    def optimizers(self):
        return [
            self.policy_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
        )