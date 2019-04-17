import tensorflow as tf
import numpy as np
from core.PPO.models import pi_model, v_model, pi_gaussian_model
from core.PPO.policy_base import PolicyBase
import tensorflow.keras.losses as kls
from utils.logger import log


class Policy_PPO_Continuous(PolicyBase):

    def __init__(self,
                 policy_params=dict(), 
                 num_actions = None):

        super().__init__(**policy_params, num_actions= num_actions)

        self.pi = pi_gaussian_model(hidden_sizes= self.hidden_sizes_pi, num_outputs= self.num_actions)
        self.v = v_model(self.hidden_sizes_v)

    def update(self, observations, actions, advs, returns, logp_t):

        for i in range(self.train_pi_iters):
            loss_pi, loss_entropy, approx_ent, kl = self.train_pi_continuous_one_step(self.pi, self.optimizer_pi, observations, actions, advs, logp_t)
            if kl > 1.5 * self.target_kl:
                log("Early stopping at step %d due to reaching max kl." %i)
                break

        # Value Update Cycle for iter steps
        for _ in range(self.train_v_iters):
            loss_v = self.train_v_one_step(self.v, self.optimizer_v, observations, returns)
            
        # Return Metrics
        return loss_pi.numpy().mean(), loss_entropy.numpy().mean(), approx_ent.numpy().mean(), kl.numpy().mean(), loss_v.numpy().mean()


    def _value_loss(self, returns, value):

        # Mean Squared Error
        loss = tf.reduce_mean((returns - value)**2)
        return loss # return kls.mean_squared_error(returns, value)


    
    def _pi_loss_continuous(self, mu, logp_old, act, adv):

        log_std = tf.Variable(name= 'log_std', initial_value= -0.5 * np.ones(self.num_actions, dtype=np.float32))
        logp = self.pi.gaussian_likelihood(act, mu, log_std)

        ratio = tf.exp(logp-logp_old)
        min_adv = tf.where(adv > 0, (1+ self.clip_ratio) * adv, (1-self.clip_ratio) * adv)

        # Policy Gradient Loss
        pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv, min_adv))

        # Entropy term
        entropy = self.entropy(log_std)
        entropy_loss = tf.reduce_mean(entropy)

        # Total Loss
        pi_loss -= self.ent_coef * entropy_loss

        # Approximated  Kullback Leibler Divergence from OLD and NEW Policy
        approx_kl = tf.reduce_mean(logp_old-logp)
        approx_ent = tf.reduce_mean(-logp) 

        return pi_loss, entropy_loss, approx_ent, approx_kl

    
    @tf.function
    def train_pi_continuous_one_step(self, model, optimizer, obs, act, adv, logp_old):

        with tf.GradientTape() as tape:

            mu = model(obs)
            pi_loss, entropy_loss, approx_ent, approx_kl  = self._pi_loss_continuous(mu, logp_old, act, adv)
            
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


    def entropy(self, log_std):

        '''
        Entropy term for more randomness which means more exploration in ppo -> 
        '''
        entropy = tf.reduce_sum(log_std + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)
        return entropy

