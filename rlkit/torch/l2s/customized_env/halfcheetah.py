import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box
import numpy as np

from rlkit.torch.l2s.customized_env.wrapper import *
import mujoco_py

class HalfcheetahWrapper(CustomizedWrapper):
    def __init__(self, env):
        super().__init__(env)
        n = self.sim.data.qpos.shape[0] + self.sim.data.qvel.shape[0]
        inf = np.array([np.inf] * n)
        self.observation_space = Box(-inf, inf)

    def is_done(self, state):
        return False

    def reset(self):
        self.sim.reset()
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        state = np.concatenate((qpos,qvel),axis=-1)
        self.set_state(state)
        return self.state_vector()

    def step(self, a):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        
        reward_ctrl = - 0.1 * np.square(a).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        state = self.state_vector()
        return state, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def get_reward(self, state, action, new_state):
        posbefore = state[0]
        posafter = new_state[0]
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (posafter - posbefore)/self.dt
        reward = reward_ctrl + reward_run
        return reward

    def predict(self, state, action):
        self.do_simulation(action, self.frame_skip)
        s = self.state_vector()
        return s
        
    def set_state(self, state):
        qpos = state[:9]
        qvel = state[9:]
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                            old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        state = np.concatenate((qpos,qvel),axis=-1)
        self.set_state(state)
        return state
                
 
    def _get_obs(self,state):
        """

        Returns:
            [type]: [description]
        """
        # return np.concatenate([
        #     self.sim.data.qpos.flat[1:],
        #     self.sim.data.qvel.flat,
        # ])
        qpos = state[:9]
        qvel = state[9:]
        return np.concatenate([
            qpos.flat[1:],
            qvel.flat
        ])
        