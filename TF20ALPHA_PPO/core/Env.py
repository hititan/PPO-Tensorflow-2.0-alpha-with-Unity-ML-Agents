from mlagents.envs import UnityEnvironment
from utils.logger import log


class UnityEnv():
    def __init__(self, env_name="", seed=0):

        self.env_name = env_name

        # Start ML Agents Environment | Without filename in editor training is started
        log("ML AGENTS INFO")
        if self.env_name=="":
            self.env = UnityEnvironment(file_name = None, seed=seed)
        else:
            self.env = UnityEnvironment(file_name = env_name, seed = seed)
        log("END ML AGENTS INFO")

        self.info = self.env.reset()[self.default_brain]
        #print(self.info)

    @property
    def get_env(self):
        return self.env

    @property
    def default_brain(self):
        return self.env.brain_names[0]

    @property
    def num_actions(self):
        brain = self.env.brains[self.default_brain]
        #shape = (None, brain.vector_action_space_size[0])
        return brain.vector_action_space_size[0]
    
    @property
    def num_obs(self):
        return self.info.vector_observations.size
    
    def reset(self):
        info = self.env.reset()[self.default_brain]
        o = info.vector_observations[0]
        r, d = 0, False
        return o, r, d

    def step(self, a):
        info = self.env.step([a])[self.default_brain] # a is int here
        r = info.rewards[0]
        d = info.local_done[0]
        o = info.vector_observations[0]
        return o, r, d

  
        
        

        