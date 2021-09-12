import gym, time
import gtimer as gt
import numpy as np
from collections import OrderedDict
import torch



from rlkit.core import logger, eval_util
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.path_builder import PathBuilder
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.policies.base import ExplorationPolicy
from rlkit.torch.l2s.policies import MakeDeterministic
from rlkit.samplers import PathSampler
from rlkit.torch.l2s.learn_env_sac import LearnEnv
from rlkit.torch.l2s.dual_env import DualEnv
from rlkit.samplers.l2s_sampler import L2SPathSampler
from rlkit.torch.l2s.utils import normalize_obs,normalize_acts

class L2SBase():
    """
    A base class for all learning to simulate environment
    i.e., learned Env
    """
    def __init__(self, 
        env,
        model,
        dual_action_dim,
        dual_obs_dim,
        primal_action_dim,
        primal_obs_dim,

        use_delta,
        use_robust,

        dual_env_set_train,
        dual_env_set_test,

        expert_dataset_list_train,
        expert_dataset_list_test,
        expert_traj_nums_train=10,
        expert_traj_nums_test=10,

        if_normalize_dataset=False,

        discriminator=None,

        discrete=True,
        gamma=0.99,

        num_epochs=100,
        num_steps_per_epoch=10000,
        num_steps_between_train_calls=1000,
        num_steps_per_eval=1000,
        max_path_length=1000,
        min_steps_before_training=0,

        env_observation_mean = None,
        env_observation_std = None,
        env_action_mean = None,
        env_action_std = None,

        no_terminal = True,
        exploration_policy=None,
        save_best = True,
        best_key='loss',

        freq_saving=10,
        save_algorithm=False,
        render=False,
        render_kwargs={}
    ):
        """[summary]

        Args:
            env ([type]): inner_env
            use_delta (bool, optional): [description]. Defaults to False.
            num_epochs (int, optional): [description]. Defaults to 100.
            num_steps_per_epoch (int, optional): [description]. Defaults to 10000.
            num_steps_between_train_calls (int, optional): [description]. Defaults to 1000.
            num_steps_per_eval (int, optional): [description]. Defaults to 1000.
            max_path_length (int, optional): [description]. Defaults to 1000.
            min_steps_before_training (int, optional): [description]. Defaults to 0.
            no_terminal (bool, optional): [description]. Defaults to False.
            exploration_policy ([type], optional): [description]. Defaults to None.
            best_key (str, optional): [description]. Defaults to 'loss'.
            freq_saving (int, optional): [description]. Defaults to 10.
            save_algorithm (bool, optional): [description]. Defaults to False.
            render (bool, optional): [description]. Defaults to False.
            render_kwargs (dict, optional): [description]. Defaults to {}.
        """  

        self.model = model
        eval_policy = MakeDeterministic(self.model)
        self.discriminator = discriminator

        self.dual_action_dim = dual_action_dim
        self.dual_obs_dim = dual_obs_dim
        self.primal_action_dim = primal_action_dim
        self.primal_obs_dim = primal_obs_dim

        self.use_delta=use_delta
        self.use_robust = use_robust

        self.num_epochs = num_epochs
        self.num_env_steps_per_epoch = num_steps_per_epoch
        self.num_steps_between_train_calls = num_steps_between_train_calls
        self.num_steps_per_eval = num_steps_per_eval
        self.max_path_length = max_path_length
        self.min_steps_before_training = min_steps_before_training

        self.discrete = discrete
        self.gamma=gamma

        self.no_terminal = no_terminal
        self.exploration_policy = exploration_policy
        self.save_best = save_best
        self.best_key = best_key

        self.freq_saving = freq_saving
        self.save_algorithm=save_algorithm
        self.render = render
        self.render_kwargs = render_kwargs

        self.expert_dataset_list_train = expert_dataset_list_train
        self.expert_dataset_list_test = expert_dataset_list_test
        self.expert_dataset_list_train_reward = None
        self.expert_dataset_list_test_reward = None
        self.expert_traj_nums_train = expert_traj_nums_train
        self.expert_traj_nums_test = expert_traj_nums_test

        self.env_observation_mean = env_observation_mean
        self.env_observation_std = env_observation_std
        self.env_action_mean = env_action_mean
        self.env_action_std = env_action_std

        self.if_normalize_dataset = if_normalize_dataset
        self.if_normalize_obs = False
        self.if_normalize_acts = False
        if self.env_observation_mean is not None:
            self.if_normalize_obs = True
            self.env_observation_mean = np.array(env_observation_mean)
            self.env_observation_std = np.array(env_observation_std)
        if self.env_action_mean is not None:
            self.if_normalize_acts = True
            self.env_action_mean = np.array(env_action_mean)
            self.env_action_std = np.array(env_action_std)

        if not self.if_normalize_dataset and (self.if_normalize_obs or self.if_normalize_acts):
            raise ValueError("when self.if_normalize_dataset=False, self.if_normalize_obs and self.if_normalize_acts should be False")

        if self.if_normalize_dataset and not self.if_normalize_obs and not self.if_normalize_acts:
            raise ValueError("when self.if_normalize_dataset=True, self.if_normalize_obs or self.if_normalize_acts should be True")

        self.learned_env = LearnEnv(env,
            self.model,
            use_delta = self.use_delta,
            env_observation_mean = self.env_observation_mean,
            env_observation_std = self.env_observation_std,
            env_action_mean = self.env_action_mean,
            env_action_std = self.env_action_std,    
        )
        
        self.dual_env_set_train = []
        self.dual_env_set_test = []
        for i in range(len(dual_env_set_train)):
            dual_env = DualEnv(env,
                model=dual_env_set_train[i],
                max_length=self.max_path_length,
                env_observation_mean = self.env_observation_mean,
                env_observation_std = self.env_observation_std,
                env_action_mean = self.env_action_mean,
                env_action_std = self.env_action_std,
            )
            self.dual_env_set_train.append(dual_env)
        for i in range(len(dual_env_set_test)):
            dual_env = DualEnv(env,
                model=dual_env_set_test[i],
                max_length=self.max_path_length,
                env_observation_mean = self.env_observation_mean,
                env_observation_std = self.env_observation_std,
                env_action_mean = self.env_action_mean,
                env_action_std = self.env_action_std,
            )
            self.dual_env_set_test.append(dual_env)
        self.bad_policy_num = int(max(2,len(self.dual_env_set_train)/4))
        # print(len(self.dual_env_set_train))
        # print(len(self.dual_env_set_test))
        
        self.train_sampler_list = []
        self.test_sampler_list = []
        for i in range(len(self.dual_env_set_train)):
            eval_sampler = L2SPathSampler(
                self.learned_env,
                self.dual_env_set_train[i],
                self.num_steps_per_eval,
                self.max_path_length,
                discrete = self.discrete,
                action_dim=self.primal_action_dim,
                no_terminal=self.no_terminal,
                gamma = self.gamma,
                discriminator = self.discriminator,
                env_observation_mean = self.env_observation_mean,
                env_observation_std = self.env_observation_std,
                env_action_mean = self.env_action_mean,
                env_action_std = self.env_action_std,
                render=self.render,
                render_kwargs=self.render_kwargs
            )
            self.train_sampler_list.append(eval_sampler)
        
        for i in range(len(self.dual_env_set_test)):
            eval_sampler = L2SPathSampler(
                self.learned_env,
                self.dual_env_set_test[i],
                self.num_steps_per_eval,
                self.max_path_length,
                discrete = self.discrete,
                action_dim=self.primal_action_dim,
                no_terminal=self.no_terminal,
                gamma = self.gamma,
                discriminator = self.discriminator,
                env_observation_mean = self.env_observation_mean,
                env_observation_std = self.env_observation_std,
                env_action_mean = self.env_action_mean,
                env_action_std = self.env_action_std,
                render=self.render,
                render_kwargs=self.render_kwargs
            )
            self.test_sampler_list.append(eval_sampler)

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._current_path_builder = PathBuilder()
        self._exploration_paths = []

        self.eval_statistics = None
        self.best_statistic_so_far=float('Inf')
        self.save_best_starting_from_epoch=0

    def train(self,start_epoch=0):
        if start_epoch == 0:
            params = self.get_epoch_snapshot(-1)
            logger.save_itr_params(-1, params)
        self.training_mode(False)
        self._n_env_steps_total = start_epoch * self.num_env_steps_per_epoch
        gt.reset()
        gt.set_def_unique(False)
        self.start_training(start_epoch=start_epoch)
    
    def get_rwd_from_dataset(self,is_train):
        if is_train:
            return self.expert_dataset_list_train_reward
        else:
            return self.expert_dataset_list_test_reward

    def get_stat_from_buffer_list(self,buffer_list,is_disc=False,batch_size=512):
        stat_list = []
        for i in range(len(buffer_list)):
            buffer = buffer_list[i]
            if is_disc:
                cur_stat = self.get_disc_from_buffer(buffer)
            else:
                cur_stat = self.get_rwd_from_buffer(buffer,batch_size = batch_size)
            stat_list.append(cur_stat)
        return stat_list

    def get_rwd_from_buffer(self,buffer,batch_size):
        raise NotImplementedError("the methods 'get_rwd_from_buffer' is deprecated")

    def get_disc_from_buffer(self,buffer):
        raise NotImplementedError

    def get_stat_from_sampler(self,sampler_list,is_disc=False):
        stat_list = []
        all_collect_data = []
        step_list = []
        for i in range(len(sampler_list)):
            single_sampler = sampler_list[i]
            collect_paths, rwds, disc, steps = single_sampler.obtain_samples()
            all_collect_data.append(collect_paths)
            step_list.append(steps)
            if is_disc:
                stat_list.append(disc)
            else:
                stat_list.append(rwds)
        return stat_list,all_collect_data,step_list

    def start_training(self, start_epoch=0):
        for epoch in gt.timed_for(
            range(start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self._start_epoch(epoch)
            for steps_this_epoch in range(self.num_env_steps_per_epoch):
                if self.no_terminal: terminal = False
                gt.stamp('sample')
                self._try_to_train(epoch)
                gt.stamp('train')

            gt.stamp('sample')
            self._try_to_eval(epoch)
            gt.stamp('eval')
            self._end_epoch()
    

    def _try_to_train(self, epoch):
        if self._can_train():
            self.training_mode(True)
            self._do_training(epoch)
            self._n_train_steps_total += 1
            self.training_mode(False)

    def _try_to_eval(self, epoch):
        if self._can_evaluate():
            if (epoch % self.freq_saving == 0) or (epoch + 1 >= self.num_epochs):
                if epoch + 1 >= self.num_epochs:
                    epoch = 'final'
                logger.save_extra_data(self.get_extra_data_to_save(epoch))
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)

            self.evaluate(epoch)

            logger.record_tabular(
                "Number of train calls total",
                self._n_train_steps_total,
            )
            logger.record_tabular(
                "Number of env steps total",
                self._n_env_steps_total,
            )

            times_itrs = gt.get_times().stamps.itrs
            train_time = times_itrs['train'][-1]
            sample_time = times_itrs['sample'][-1]
            compute_time = times_itrs['compute_diff'][-1]
            eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time = train_time + sample_time + eval_time + compute_time
            total_time = gt.get_times().total

            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('Compute Time (s)', compute_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Sample Time (s)', sample_time)
            logger.record_tabular('Epoch Time (s)', epoch_time)
            logger.record_tabular('Total Train Time (s)', total_time)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")
    
    def evaluate(self, epoch):
        """
        Evaluate the policy, e.g. save/print progress.
        :param epoch:
        :return:
        """
        statistics = OrderedDict()
        try:
            statistics.update(self.eval_statistics)
            self.eval_statistics = None
        except:
            print('No Stats to Eval')

        logger.log("Collecting samples for evaluation")

        for key, value in statistics.items():
            logger.record_tabular(key, value)
        
        best_statistic = statistics[self.best_key]
        if best_statistic < self.best_statistic_so_far:
            self.best_statistic_so_far = best_statistic
            if self.save_best and epoch >= self.save_best_starting_from_epoch:
                data_to_save = {
                    'epoch': epoch,
                    'statistics': statistics
                }
                data_to_save.update(self.get_epoch_snapshot(epoch))
                logger.save_extra_data(data_to_save, 'best.pkl')
                print('\n\nSAVED BEST\n\n')
    
    def _get_action_and_info(self, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
        return self.exploration_policy.get_action(
            observation,
        )
    
    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)
    
    def _end_epoch(self):
        self.eval_statistics = None
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()
    
    def get_epoch_snapshot(self, epoch):
        snapshot = dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy
        )
        return snapshot

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    @property
    def networks(self):
        """
        Used in many settings such as moving to devices
        """
        pass

    def to(self,device):
        for net in self.networks:
            net.to(device)
        
    def _can_train(self):
        pass

    def _can_evaluate(self):
        pass

    def set_dataset_info(self, train_policy_steps_mean,train_policy_rwd_mean,test_policy_steps_mean,test_policy_rwd_mean):
        self.expert_dataset_list_train_reward = np.array(train_policy_rwd_mean)
        self.expert_dataset_list_test_reward = np.array(test_policy_rwd_mean)
        self.expert_dataset_list_train_steps = np.array(train_policy_steps_mean)
        self.expert_dataset_list_test_steps = np.array(test_policy_steps_mean)

    def show_para(self,params):
        for i in params:
            if torch.any(torch.isnan(i)):
                print(i)

    def show_all_para(self,params):
        for i in params:
                print(i)

    def normalize_batch(self,batch):
        if self.if_normalize_obs:
            batch['observations'] = normalize_obs(batch['observations'],self.env_observation_mean,self.env_observation_std)
            batch['next_observations'] = normalize_obs(batch['next_observations'],self.env_observation_mean,self.env_observation_std)
        if self.if_normalize_acts:
            batch['actions'] = normalize_obs(batch['actions'],self.env_action_mean,self.env_action_std)
        return batch