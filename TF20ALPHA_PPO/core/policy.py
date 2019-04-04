import tensorflow as tf
import numpy as np
from core.models import pi_model, v_model
import tensorflow.keras.losses as kls
from utils.logger import log


class Policy_PPO():

    def __init__(self,
                 lr_v=0.001,
                 lr_pi=0.001,
                 hidden_sizes_pi=(64,64),
                 hidden_sizes_v=(64,64),
                 train_pi_iters=80,
                 train_v_iters=80,
                 clip_ratio=0.2,
                 target_kl=0.01,
                 num_actions=None):

        self.lr_v = lr_v
        self.lr_pi = lr_pi
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.num_actions = num_actions

        self.pi = pi_model(hidden_sizes_pi, num_actions)
        self.v = v_model(hidden_sizes_v)

        self.optimizer_pi = tf.keras.optimizers.Adam(lr=lr_pi)
        self.optimizer_v = tf.keras.optimizers.Adam(lr=lr_v)

        self.loss_metric = tf.keras.metrics.Mean(name='train_loss')


    def update(self, observations, actions, advs, returns, logp_t):
        self.loss_metric.reset_states()
        for i in range(self.train_pi_iters):
            loss_pi, kl = self.train_pi_one_step(self.pi, self.optimizer_pi, observations, actions, advs, logp_t)
            #print('Loss_Pi: {:.2f}'.format(loss_pi.numpy().mean()))
            #print('KL     :' + str(kl.numpy().mean()))
            if kl > 1.5 * self.target_kl:
                log("Early stopping at step %d due to reaching max kl." %i)
                break
        #mean_loss = self.loss_metric.result()
        #log('Loss: {:.3f}'.format(mean_loss))
        #log('KL  : {:.3f}'.format(kl.numpy().mean()))
        for _ in range(self.train_v_iters):
            loss_v = self.train_v_one_step(self.v, self.optimizer_v, observations, returns)
        #log('V-Loss: {:.3f}'.format(loss_v.numpy().mean())) 
    
    def _value_loss(self, returns, value):
        return kls.mean_squared_error(returns, value)

    def _pi_loss(self, logits, logp_old, act, adv):
        logp = tf.reduce_sum(tf.one_hot(act,self.num_actions)*tf.nn.log_softmax(logits), axis=1)
        ratio = tf.exp(logp-logp_old)
        min_adv = tf.where(adv > 0, (1+ self.clip_ratio) * adv, (1-self.clip_ratio) * adv)
        pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv, min_adv))
        approx_kl = tf.reduce_mean(logp_old-logp)
        return pi_loss, approx_kl
    
    
    @tf.function
    def train_pi_one_step(self, model, optimizer, obs, act, adv, logp_old):
        with tf.GradientTape() as tape:
            logits = model(obs)
            pi_loss, approx_kl = self._pi_loss(logits, logp_old, act, adv)
            
        grads = tape.gradient(pi_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        self.loss_metric.update_state(pi_loss)
        return pi_loss, approx_kl


    @tf.function()
    def train_v_one_step(self, model, optimizer_v, obs, returns):
        with tf.GradientTape() as tape:
            values = model(obs)
            v_loss = self._value_loss(returns, values)
        grads = tape.gradient(v_loss, model.trainable_variables)
        optimizer_v.apply_gradients(zip(grads, model.trainable_variables))
        return v_loss

