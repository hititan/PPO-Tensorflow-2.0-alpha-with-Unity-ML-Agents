import tensorflow as tf
import numpy as np
import time
from utils.logger import log
from pprint import pprint
from mlagents.envs import UnityEnvironment
from core.buffer import Buffer_PPO
from core.policy import Policy_PPO


class Trainer_PPO:
    def __init__(self,
                 env=UnityEnvironment,
                 epochs=10,
                 steps_per_epoch=1000,
                 max_episode_length=1000,
                 gamma=0.99,
                 lam=0.97,
                 seed=0,
                 policy_params=dict(),
                 **kwargs):

        self.env = env
        self.epochs= epochs
        self.steps_per_epoch = steps_per_epoch
        self.max_episode_length =max_episode_length
        self.gamma = gamma
        self.lam = lam
        self.seed= seed
        self.policy_params = policy_params

        log("Policy Parameters")
        pprint(policy_params, indent=5, width=10)

        self.buffer = Buffer_PPO(steps_per_epoch, gamma=gamma, lam=lam)

        log(policy_params)
        self.agent = Policy_PPO(**policy_params, num_actions= self.num_actions)

        self.rew_metric = tf.keras.metrics.Mean(name='train_loss')
        self.summary_writer = tf.summary.create_file_writer('./tmp/summaries')


    def start(self):
        log("Starting Trainer ...")
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        self.train()


    # Main training loop
    def train(self):

        start_time = time.time()
        
        info = self.env.reset()[self.default_brain]
        o= info.vector_observations[0]
        r, d, ep_ret, ep_len = 0, False, 0, 0
         
        for epoch in range(self.epochs):

            self.rew_metric.reset_states()

            for step in range(self.steps_per_epoch):

                a, logp_t = self.agent.pi.get_action_logp(o[None, :])
                v_t = self.agent.v.get_value(o[None, :])           
                 
                self.buffer.store(o, a, r, v_t, logp_t)
                
                # make step in env
                info = self.env.step([a])[self.default_brain] # a is int here

                r = info.rewards[0]
                d = info.local_done[0]
                o = info.vector_observations[0]
                  
                ep_ret += r
                ep_len += 1

                terminal =  d or (ep_len == self.max_episode_length)

                if terminal or (step == self.steps_per_epoch-1):
                    if not terminal:
                        log('Warning: trajectory was cut off by epoch at %d steps.' %(ep_len))

                    last_val = r if d else self.agent.v.get_value(o[None, :])
                    self.buffer.finish_path(last_val)

                    if terminal:
                        self.rew_metric.update_state(ep_ret)
                        # self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                                
                    info = self.env.reset()[self.default_brain]
                    o = info.vector_observations[0]
                    r, d, ep_ret, ep_len = 0, False, 0, 0
      
            #Update via PPO and Logging
            obs, act, adv, ret, logp_old = self.buffer.get()
            self.agent.update(obs,act,adv, ret, logp_old)
            
            mean_rew = self.rew_metric.result()
            log('Mean Reward: {:.3f}'.format(mean_rew))

            with self.summary_writer.as_default():
                tf.summary.scalar('Mean Reward', mean_rew, step=epoch)

    
    @property
    def default_brain(self):
        return self.env.brain_names[0]

    @property
    def num_actions(self):
        brain = self.env.brains[self.default_brain]
        info = self.env.reset()[self.default_brain]
        return brain.vector_action_space_size[0]