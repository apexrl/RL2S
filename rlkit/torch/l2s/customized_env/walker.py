import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box
import numpy as np

from rlkit.torch.l2s.customized_env.wrapper import *
import mujoco_py



class WalkerWrapper(CustomizedWrapper):
    def __init__(self, env):
        super().__init__(env)
        n = self.sim.data.qpos.shape[0] + self.sim.data.qvel.shape[0]
        inf = np.array([np.inf] * n)
        self.observation_space = Box(-inf, inf)

    def is_done(self, state):
        posafter, height, ang = state[0:3]
        done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        return bool(done)

    def reset(self):
        self.sim.reset()
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        state = np.concatenate((qpos,qvel),axis=-1)
        self.set_state(state)
        return self.state_vector()

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        state = self.state_vector()
        return state, reward, done, {}

    def get_reward(self, state, action, new_state):
        posbefore = state[0]
        posafter, height, ang = new_state[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(action).sum()
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
                
 
    def _get_obs(self, state):
        """[used in true hopper env]

        Returns:
            [type]: [description]
        """
        qpos = state[:9]
        qvel = state[9:]
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()
        # return np.concatenate([
        #     qpos.flat[1:],
        #     np.clip(qvel.flat, -10, 10)
        # ])