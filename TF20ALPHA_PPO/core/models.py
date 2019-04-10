import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as kl

class ProbabilityDistribution(tf.keras.Model):

    def call(self, logits):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
    

class pi_model(tf.keras.Model):

    def __init__(self, hidden_sizes_pi=(32,32), num_actions=None):

        super().__init__('pi_model')

        self.num_actions = num_actions
        self.hidden_pi_layers = tf.keras.Sequential([kl.Dense(h, activation='relu') for h in hidden_sizes_pi])
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()

    def call(self, inputs):

        tensor_input = tf.convert_to_tensor(inputs)
        hidden_logs = self.hidden_pi_layers(tensor_input)
        return self.logits(hidden_logs)
    
    def get_action_logp(self, obs):

        logits = self.predict(obs)
        logp_all = tf.nn.log_softmax(logits)
        action = self.dist.predict(logits)
        logp_t = tf.reduce_sum(tf.one_hot(action, depth=self.num_actions) * logp_all, axis=1)
        return tf.squeeze(action, axis=-1), np.squeeze(logp_t, axis=-1)


class v_model(tf.keras.Model):

    def __init__(self, hidden_sizes_v=(32,32)):

        super().__init__('v_model')

        self.hidden_v_layers = tf.keras.Sequential([kl.Dense(h, activation='relu') for h in hidden_sizes_v])
        self.value= kl.Dense(1, name='value')

    def call(self, inputs):

        tensor_input = tf.convert_to_tensor(inputs)
        hidden_vals = self.hidden_v_layers(tensor_input)
        return  self.value(hidden_vals)
    
    def get_value(self, obs):

        value = self.predict(obs)
        return  np.squeeze(value, axis=-1)