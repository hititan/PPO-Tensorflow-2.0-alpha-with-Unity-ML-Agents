import tensorflow as tf
import numpy as np
from utils.logger import log


class PolicyBase:

    def __init__(self,
                 lr_v=0.001,
                 lr_pi=0.001,
                 hidden_sizes_pi=(64, 64),
                 hidden_sizes_v=(64, 64),
                 train_pi_iters=80,
                 train_v_iters=80,
                 clip_ratio=0.2,
                 target_kl=0.01,
                 ent_coef=0,
                 use_sil=False,
                 sil_iters=4,
                 num_actions=None,
                 **kwargs):
        
        # Arguments from config policy parameters
        self.lr_v = lr_v
        self.lr_pi = lr_pi
        self.hidden_sizes_pi = hidden_sizes_pi
        self.hidden_sizes_v = hidden_sizes_v
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.ent_coef = ent_coef

        # Arguments from config for SIL
        self.use_sil = use_sil
        self.sil_iters = sil_iters

        # Additional Arguments
        self.num_actions = num_actions

        # init pi, v and optimizers
        self.pi = None
        self.v = None

        self.optimizer_pi = tf.keras.optimizers.Adam(lr= self.lr_pi)
        self.optimizer_v = tf.keras.optimizers.Adam(lr= self.lr_v)


    def save(self):

        log('Saving Model ...')
        self.pi.save_weights('./tmp/ckpts/pi', save_format='tf')
        self.v.save_weights('./tmp/ckpts/v', save_format='tf')

    def load(self):
        
        log('Loading Model ...')
        self.pi.load_weights('./tmp/ckpts/pi')
        self.v.load_weights('./tmp/ckpts/v')