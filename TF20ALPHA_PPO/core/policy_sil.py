import numpy as np
import tensorflow as tf
from utils.logger import log



class SIL:

    def __init__(self, pi, v, optimizer_pi, optimizer_v, num_actions):

        self.pi = pi
        self.v = v
        self.optimizer_pi = optimizer_pi
        self.optimizer_v = optimizer_v

        self.num_actions = num_actions

    
    def train_sil_policy_one_step(self, obs, act, R): # is_weights):

        with tf.GradientTape() as tape:

            logits = self.pi(obs)
            V_Pred = self.v.get_value(obs)

            pi_loss, adv  = self.sil_policy_loss(logits, act, R, V_Pred) 
            
        grads = tape.gradient(pi_loss, self.pi.trainable_variables)
        self.optimizer_pi.apply_gradients(zip(grads, self.pi.trainable_variables))

        return pi_loss, adv

    
    def train_sil_value_one_step(self, obs, R): # is_weights):

        with tf.GradientTape() as tape:

            V_Pred = self.v(obs)
            v_loss = self.sil_value_loss(R, V_Pred) 

        grads = tape.gradient(v_loss, self.v.trainable_variables)
        self.optimizer_v.apply_gradients(zip(grads, self.v.trainable_variables))

        return v_loss

    def train_sil_one_step(self, obs, act, R): # is_weights):

        pi_loss, adv = self.train_sil_policy_one_step(obs, act, R)
        v_loss = self.train_sil_value_one_step(obs, R)

        return pi_loss, adv, v_loss


    def sil_policy_loss(self, logits, act, R, V_Pred):
        '''
            sil_policy_loss = -log_prob * max(R - V_Pred, 0)
            sil_val_loss = 0.5 * max(R - V_Pred, 0) ** 2
            Called on Batch sampled from PER

        '''
        logp = self.logp(logits, act)

        clipped_advs = tf.math.maximum(R - tf.squeeze(V_Pred), 0)

        sil_policy_loss = tf.reduce_mean(-logp * clipped_advs)

        return sil_policy_loss, clipped_advs
    

    def sil_value_loss(self, R, V_Pred):
        '''
            sil_policy_loss = -log_prob * max(R - V_Pred, 0)
            sil_val_loss = 0.5 * max(R - V_Pred, 0) ** 2
            Called on Batch sampled from PER

        '''
        clipped_advs = tf.math.maximum(R - tf.squeeze(V_Pred), 0)
        sil_val_loss = 0.01 * tf.reduce_mean(clipped_advs**2)

        return sil_val_loss

    
    def logp(self, logits, act):

        # make logp for policy_gradient update
        logp_all = tf.nn.log_softmax(logits)
        logp = tf.reduce_sum( tf.one_hot(act, self.num_actions) * logp_all, axis=1)

        return logp
