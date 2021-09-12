import numpy as np
from rlkit.data_management.path_builder import PathBuilder
import torch
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.l2s.utils import unnormalize_obs, normalize_acts,normalize_obs, get_raw_ob_from_state


def l2s_rollout(
    env, 
    policy,
    max_path_length,
    discrete,
    action_dim,
    no_terminal=False,
    use_gail=False,
    gamma=0.99,
    alg_type="sac",
    discriminator=None,
    vf=None,
    lam=0.9,
    env_observation_mean = None,
    env_observation_std = None,
    env_action_mean = None,
    env_action_std = None,
    render=False,
    render_kwargs={},
):
    """[summary]

    Args:
        env ([type]): primal env , not dual_env, learned dynamic model
        policy ([type]): primal policy , not dual_policy
        max_path_length ([type]): [description]
        no_terminal (bool, optional): [description]. Defaults to False.
        gamma (float, optional): [description]. Defaults to 0.99.
        lam (float, optional): [description]. Defaults to 0.9,used when TRPO
        discriminator ([type], optional): [description]. Defaults to None.
        vf ([type], optional): [description]. Defaults to None.
        render (bool, optional): [description]. Defaults to False.
        render_kwargs (dict, optional): [description]. Defaults to {}.

    Returns:
        [type]: [description]
    """
    if alg_type=="trpo" and vf is None:
        raise ValueError("You should provide vf when choose trpo sampler")
    path_builder = PathBuilder()
    observation = env.reset() # normalized obs
    ep_rwds = 0
    ep_disc = 0
    c=1
    cur_step = 0
    if_normalize_obs=False
    if_normalize_acts=False
    if env_observation_mean is not None:
        if_normalize_obs = True
    if env_action_mean is not None:
        if_normalize_acts = True
    #terminal = False
    with torch.no_grad():
        for _ in range(max_path_length):
            action, agent_info = policy.get_action(observation) #policy is dual env,action is normalized in dual_env

            # if if_normalize_obs:
            #     unnormal_observation = unnormalize_obs(observation,env_observation_mean,env_observation_std)
            #     action, agent_info = policy.get_action(unnormal_observation)
            # else:
            #     action, agent_info = policy.get_action(observation)
            
            # if if_normalize_acts:
            #     action = normalize_acts(action,env_action_mean,env_action_std)

            if render: env.render(**render_kwargs)
            
            next_ob, reward, terminal, env_info = env.step(action) # env already consider the delta

            if no_terminal: terminal = False

            # if _<=2:
            #     print(observation)

            path_builder.add_all(
                observations=observation,
                actions=action,
                rewards=np.array([reward]),
                next_observations=next_ob,
                terminals=np.array([terminal]),
                absorbing=np.array([0., 0.]),
                agent_info=agent_info,
                env_info=env_info,
            )
            cur_step+=1
            observation = next_ob
            if terminal: break
        observations = np.array(path_builder["observations"])
        actions = np.array(path_builder["actions"])
        next_observations = np.array(path_builder["next_observations"])
        rewards = np.squeeze(np.array(path_builder["rewards"]))
        if discrete:
            actions = np.eye(action_dim)[actions]
            path_builder["actions"] = actions

        if use_gail:
            padding = np.ones((observations.shape[0],1),dtype=np.float)
            actions = np.concatenate([actions,padding],axis=-1)

        observation_tensor = torch.from_numpy(observations).to(ptu.device).double()
        action_tensor = torch.from_numpy(actions).to(ptu.device).double()
        next_observations_tensor = torch.from_numpy(next_observations).to(ptu.device).double()
        
        gamma_array = np.power(gamma,np.arange(0,cur_step))

        # print(observation_tensor.type())
        # print(action_tensor.type())
        # print(next_observations_tensor.type())
        
        if discriminator is not None:
            sa_tensor = torch.cat([observation_tensor,action_tensor,next_observations_tensor],dim=-1)
            sa_tensor = sa_tensor.float()
            _,disc_value = discriminator(sa_tensor)
            disc_value = disc_value.detach().cpu().numpy()
            disc_value =np.squeeze(disc_value)
            ep_disc = np.sum(disc_value*gamma_array)
        else:
            ep_disc = None

        ep_rwds = np.sum(rewards*gamma_array)
        #print(path_builder['observations'][0:2])
        
        if alg_type=="trpo":
            value_predict = vf(observation_tensor)
            value_predict = value_predict.detach().cpu().numpy()
            value_predict = np.squeeze(value_predict)

            last_obs = torch.from_numpy(np.array(next_ob))
            last_obs = torch.unsqueeze(last_obs,dim=0)
            last_value = vf(last_obs)
            last_value = torch.squeeze(last_value).detach().cpu().numpy()[0]
            last_value = float(last_value)

            lastgaelam = 0
            gae=np.empty((len(observations),1),'float32')
            ret = np.empty((len(observations),1),'float32')
            if terminal:
                cur_ret = 0
            else:
                cur_ret = last_value
            
            nonterminal=1
            for step in reversed(range(len(observations))):
                cur_ret = rewards[step] + gamma*cur_ret
                ret[step][0] = cur_ret
                if step==len(observations)-1:
                    if terminal:
                        delta = rewards[step] -value_predict[step]
                        gae[step][0] = lastgaelam =delta
                    else:
                        delta = rewards[step] + last_value-value_predict[step]
                        gae[step][0] = lastgaelam =delta+gamma*lam*lastgaelam
                    continue
                delta = rewards[step] + value_predict[step+1]*nonterminal-value_predict[step]
                gae[step][0] = lastgaelam =delta+gamma*lam*nonterminal*lastgaelam
            path_builder["vpreds"] = list(value_predict)
            path_builder["gae"]=list(gae)
            path_builder["ret"]=list(ret)
    return path_builder, ep_rwds, ep_disc


def rolloutSimMujoco(
    env,
    policy,
    max_path_length,
    steps_after_done = 0,
    no_terminal=False,
    render=False,
    render_kwargs={},
):
    path_builder = PathBuilder()
    observation = env.reset()
    last_terminal=False

    for _ in range(max_path_length+steps_after_done):

        raw_observation = env._get_obs(observation)

        #raw_observation is used in primal policy
        action, agent_info = policy.get_action(raw_observation)

        #action, agent_info = policy.get_action(observation)

        if render: env.render(**render_kwargs)

        next_ob, reward, terminal, env_info = env.step(action)
        
        if last_terminal:
            terminal = True
            
        if no_terminal: terminal = False

        last_terminal = terminal

        path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=np.array([reward]),
            next_observations=next_ob,
            terminals=np.array([terminal]),
            absorbing=np.array([0., 0.]),
            agent_info=agent_info,
            env_info=env_info,
        )

        observation = next_ob
        if terminal: 
            if steps_after_done<=0:
                break
            else:
                steps_after_done-=1
    return path_builder


class L2SPathSampler():
    def __init__(
        self,
        env,
        policy,
        num_steps,
        max_path_length,
        discrete,
        action_dim,
        no_terminal=False,
        use_gail=False,
        gamma=0.99,
        alg_type="sac",
        discriminator=None,
        vf=None,
        lam=0.9,
        env_observation_mean = None,
        env_observation_std = None,
        env_action_mean = None,
        env_action_std = None,
        render=False,
        render_kwargs={}
    ):
        """
        When obtain_samples is called, the path sampler will generates the
        minimum number of rollouts such that at least num_steps timesteps
        have been sampled
        """
        if alg_type=="trpo" and vf is None:
            raise ValueError("You should provide vf when choose trpo sampler")
        self.env = env
        self.policy = policy
        self.num_steps = num_steps
        self.max_path_length = max_path_length
        self.discrete = discrete
        self.action_dim =action_dim
        self.no_terminal = no_terminal
        self.use_gail = use_gail
        self.render = render
        self.render_kwargs = render_kwargs
        self.alg_type=alg_type
        self.discriminator = discriminator
        self.vf = vf
        self.gamma = gamma
        self.lam = lam
        self.env_observation_mean = env_observation_mean
        self.env_observation_std = env_observation_std
        self.env_action_mean = env_action_mean
        self.env_action_std = env_action_std
    

    def obtain_samples(self, num_steps=None):
        paths = []
        total_steps = 0
        rwd_list = []
        disc_list = []
        step_list = []
        if num_steps is None:
            num_steps = self.num_steps
        while total_steps < num_steps:
            new_path,ep_rwds, ep_disc = l2s_rollout(
                self.env,
                self.policy,
                self.max_path_length,
                self.discrete,
                self.action_dim,
                no_terminal=self.no_terminal,
                use_gail=self.use_gail,
                gamma =self.gamma,
                alg_type=self.alg_type,
                discriminator=self.discriminator,
                vf = self.vf,
                lam = self.lam,
                env_observation_mean = self.env_observation_mean,
                env_observation_std = self.env_observation_std,
                env_action_mean = self.env_action_mean,
                env_action_std = self.env_action_std,
                render=self.render,
                render_kwargs=self.render_kwargs
            )
            rwd_list.append(ep_rwds)
            disc_list.append(ep_disc)
            paths.append(new_path)
            step_list.append(len(new_path['rewards']))
            total_steps += len(new_path['rewards'])
        mean_rwds = np.mean(rwd_list)
        if self.discriminator is not None:
            mean_disc = np.mean(disc_list)
        else:
            mean_disc = None
        mean_steps = np.mean(step_list)
        # print(paths[0]['observations'][0:2])
        # exit(1)
        return paths,mean_rwds,mean_disc,mean_steps
