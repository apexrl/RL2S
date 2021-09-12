import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box
import numpy as np

from rlkit.torch.l2s.customized_env.wrapper import *
import mujoco_py



class AntWrapper(CustomizedWrapper):
    def __init__(self, env):
        super().__init__(env)
        n = self.sim.data.qpos.shape[0] + self.sim.data.qvel.shape[0] + 85
        inf = np.array([np.inf] * n)
        self.observation_space = Box(-inf, inf)

    def is_done(self, state):
        true_state = state[1:30]
        notdone = np.isfinite(true_state).all() and true_state[2] >= 0.2 and true_state[2] <= 1.0
        done = not notdone
        return bool(done)

    def reset(self):
        self.sim.reset()
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        state = np.concatenate((qpos,qvel),axis=-1)
        self.set_state(state)
        xposbefore = self.get_body_com("torso")[0]
        new_state = np.concatenate([[xposbefore],state,self.sim.data.cfrc_ext.flat])
        return new_state

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        new_state = np.concatenate([[xposafter],state,self.sim.data.cfrc_ext.flat])
        return new_state, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def get_reward(self, state, action, new_state):
        posbefore = state[0]
        posafter = new_state[0]
        alive_bonus = 1.0
        forward_reward = (posafter - posbefore)/self.dt
        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(new_state[30:], -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        return reward

    def predict(self, state, action):
        self.do_simulation(action, self.frame_skip)
        s = self.state_vector()
        xposbefore = self.get_body_com("torso")[0]
        new_state = np.concatenate([[xposbefore],s,self.sim.data.cfrc_ext.flat])
        return new_state
        
    def set_state(self, state):
        qpos = state[:15]
        qvel = state[15:]
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                            old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def reset_model(self):
        self.sim.reset()
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        state = np.concatenate((qpos,qvel),axis=-1)
        self.set_state(state)
        xposbefore = self.get_body_com("torso")[0]
        new_state = np.concatenate([[xposbefore],state,self.sim.data.cfrc_ext.flat])
        return new_state
                
 
    def _get_obs(self, state):
        """[used in true ant env]

        Returns:
            [type]: [description]
        """
        torso_place = state[0]
        qpos = state[1:16]
        qvel = state[16:30]
        cfrc_ext = state[30:]
        return np.concatenate([qpos[2:],qvel,np.clip(cfrc_ext,-1,1).flat])
        # return np.concatenate([
        #     qpos.flat[1:],
        #     np.clip(qvel.flat, -10, 10)
        # ])