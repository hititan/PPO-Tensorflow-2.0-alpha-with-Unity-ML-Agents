import tensorflow as tf
import numpy as np
from core.models import pi_model, v_model, pi_gaussian_model
from core.policy_base import PolicyBase
import tensorflow.keras.losses as kls
from utils.logger import log


class Policy_PPO_Categorical(PolicyBase):

    def __init__(self,
                 policy_params=dict(), 
                 num_actions = None):

        super().__init__(**policy_params, num_actions= num_actions)

        self.pi = pi_model(self.hidden_sizes_pi, self.num_actions)
        self.v = v_model(self.hidden_sizes_v)


    def update(self, observations, actions, advs, returns, logp_t):

        # Policy Update Cycle for iter steps
        for i in range(self.train_pi_iters):
            loss_pi, loss_entropy, approx_ent, kl = self.train_pi_one_step(self.pi, self.optimizer_pi, observations, actions, advs, logp_t)
            if kl > 1.5 * self.target_kl:
                log("Early stopping at step %d due to reaching max kl." %i)
                break

        # Value Update Cycle for iter steps
        for _ in range(self.train_v_iters):
            loss_v = self.train_v_one_step(self.v, self.optimizer_v, observations, returns)
            
        # Return Metrics
        return loss_pi.numpy().mean(), loss_entropy.numpy().mean(), approx_ent.numpy().mean(), kl.numpy().mean(), loss_v.numpy().mean()
        
    def update_self_imitation(self, obs, acts, R, is_weights):

        for i in range(4):
            loss_pi, adv = self.train_pi_imitation_one_step(self.pi, self.optimizer_pi, obs, acts, R, is_weights)
            loss_v = self.train_v_imitation_one_step (self.v, self.optimizer_v, obs, R, is_weights)

        return adv, loss_pi, loss_v


    def _value_loss(self, returns, value):

        # Mean Squared Error
        loss = tf.reduce_mean((returns - value)**2)
        return loss 


    def _pi_loss(self, logits, logp_old, act, adv):

        # PPO Objective 
        logp_all = tf.nn.log_softmax(logits)
        logp = tf.reduce_sum( tf.one_hot(act, self.num_actions) * logp_all, axis=1)
        ratio = tf.exp(logp-logp_old)
        min_adv = tf.where(adv > 0, (1+ self.clip_ratio) * adv, (1-self.clip_ratio) * adv)

        # Policy Gradient Loss
        pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv, min_adv))

        # Entropy loss 
        # entropy_loss_2 = tf.reduce_mean(self.entropy2(logits)) # calculated with Cross Entropy over logits with itself ??
        entropy_loss = tf.reduce_mean(self.entropy(logits))

        # Total Loss
        pi_loss -= self.ent_coef * entropy_loss

        # Approximated  Kullback Leibler Divergence from OLD and NEW Policy
        approx_kl = tf.reduce_mean(logp_old-logp)
        approx_ent = tf.reduce_mean(-logp) 

        return pi_loss, entropy_loss, approx_ent, approx_kl

    
    @tf.function
    def train_pi_one_step(self, model, optimizer, obs, act, adv, logp_old):

        with tf.GradientTape() as tape:

            logits = model(obs)
            pi_loss, entropy_loss, approx_ent, approx_kl  = self._pi_loss(logits, logp_old, act, adv)
            
        grads = tape.gradient(pi_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return pi_loss, entropy_loss, approx_ent, approx_kl


    @tf.function()
    def train_v_one_step(self, model, optimizer_v, obs, returns):

        with tf.GradientTape() as tape:

            values = model(obs)
            v_loss = self._value_loss(returns, values)

        grads = tape.gradient(v_loss, model.trainable_variables)
        optimizer_v.apply_gradients(zip(grads, model.trainable_variables))

        return v_loss


    def entropy(self, logits):

        '''
        Entropy term for more randomness which means more exploration in ppo -> 
        
        Due to machine precission error -> 
        entropy = - tf.reduce_sum (logp_all * tf.log(logp_all) + 1E-12, axis=-1, keepdims=True) 
        cannot be calculated this way
        '''
        
        a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        exp_a0 = tf.exp(a0)
        z0 = tf.reduce_sum(exp_a0, axis=-1, keepdims=True)
        p0 = exp_a0 / z0
        entropy = tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)

        return entropy

    def entropy2(self, logits):

        entropy = kls.categorical_crossentropy(logits, logits, from_logits=True)
        return entropy



    # ------------------------------------------- Self Imitation
    
    def train_pi_imitation_one_step(self, model, optimizer, obs, act, R, is_weights):

        with tf.GradientTape() as tape:

            logits = model(obs)
            Model_V = self.v.get_value(obs)

            pi_loss, adv  = self._self_imitation_pg_loss(logits, act, R, Model_V, is_weights)
            
        grads = tape.gradient(pi_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return pi_loss, adv

    
    def train_v_imitation_one_step(self, model, optimizer_v, obs, R, is_weights):

        with tf.GradientTape() as tape:

            Model_V = model(obs)
            v_loss = self._self_imitation_v_loss(R, Model_V, is_weights)

        grads = tape.gradient(v_loss, model.trainable_variables)
        optimizer_v.apply_gradients(zip(grads, model.trainable_variables))

        return v_loss


    def _self_imitation_pg_loss(self, logits, act_imitation, R, Model_V, is_weights, min_batch_size=64, clip = 1):

        # make a maks of valid samples where (R - V > 0) --> (0,1,0,0,1,...)
        mask = tf.where(R - tf.squeeze(Model_V) > 0.0, tf.ones_like(R), tf.zeros_like(R))
        num_valid_samples = tf.reduce_sum(mask)
        num_samples = tf.maximum(num_valid_samples, min_batch_size)

        # make logp for policy_gradient update
        logp_all = tf.nn.log_softmax(logits)
        logp = tf.reduce_sum( tf.one_hot(act_imitation, self.num_actions) * logp_all, axis=1)

        # calculate adv = max [(R-V), 0]
        # clipped advantage as priority for PER
        adv = tf.stop_gradient(tf.clip_by_value(R - tf.squeeze(Model_V), 0.0, clip))
        mean_adv = tf.reduce_sum(adv) / num_samples

        loss = -tf.reduce_sum(is_weights * adv * logp) / num_samples

        entropy = tf.reduce_sum(is_weights * self.entropy(logits) * mask) / num_samples

        loss -= entropy * 0.01 # w_entropy_coeff

        return loss, adv

    def _self_imitation_v_loss(self, R, Model_V, is_weights, min_batch_size=64, clip = 1):

        mask = tf.where(R - tf.squeeze(Model_V) > 0.0, tf.ones_like(R), tf.zeros_like(R))
        num_valid_samples = tf.reduce_sum(mask)
        num_samples = tf.maximum(num_valid_samples, min_batch_size)

        v_target = R
        v_estimate = tf.squeeze(Model_V)

        delta = tf.clip_by_value(v_estimate - v_target, -clip, 0) * mask
        loss = tf.reduce_sum(is_weights * v_estimate * tf.stop_gradient(delta)) / num_samples
        loss = 0.5 * loss * 0.01

        return loss




