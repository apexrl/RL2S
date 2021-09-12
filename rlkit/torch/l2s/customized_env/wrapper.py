import gym
class CustomizedWrapper(gym.Wrapper):

    def is_done(self, state):
        raise RuntimeError("Not implemented")
    
    def predict(self, state, action):
        raise RuntimeError("Not implemented")
    
    # def set_state(self, state):
    #     raise RuntimeError("Not implemented")

    def get_reward(self, state, action, new_state):
        raise RuntimeError("Not implemented")
    
    def absorbing_state(self):
        raise RuntimeError("Not implemented")
