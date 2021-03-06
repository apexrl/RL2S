import numpy as np
from rlkit.data_management.path_builder import PathBuilder

def rollout(
    env,
    policy,
    max_path_length,
    no_terminal=False,
    render=False,
    render_kwargs={},
):
    path_builder = PathBuilder()
    observation = env.reset()

    for _ in range(max_path_length):
        action, agent_info = policy.get_action(observation)
        if np.isnan(action).any():
            print(observation)
            print('NAN actions')
            return []
            #exit(1)
        if render: env.render(**render_kwargs)

        next_ob, reward, terminal, env_info = env.step(action)
        if no_terminal: terminal = False

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
        if terminal: break
    return path_builder


class PathSampler():
    def __init__(
        self,
        env,
        policy,
        num_steps,
        max_path_length,
        no_terminal=False,
        render=False,
        render_kwargs={}
    ):
        """
        When obtain_samples is called, the path sampler will generates the
        minimum number of rollouts such that at least num_steps timesteps
        have been sampled
        """
        self.env = env
        self.policy = policy
        self.num_steps = num_steps
        self.max_path_length = max_path_length
        self.no_terminal = no_terminal
        self.render = render
        self.render_kwargs = render_kwargs
    

    def obtain_samples(self, num_steps=None):
        paths = []
        total_steps = 0
        if num_steps is None:
            num_steps = self.num_steps
        while total_steps < num_steps:
            new_path = rollout(
                self.env,
                self.policy,
                self.max_path_length,
                no_terminal=self.no_terminal,
                render=self.render,
                render_kwargs=self.render_kwargs
            )
            if(len(new_path)!=0):
                paths.append(new_path)
                total_steps += len(new_path['rewards'])
        return paths
