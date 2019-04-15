import random
import numpy as np
from core.buffers.SumTree import SumTree

class Memory:

    e = 0.01
    a = 0.6
    beta =0.4
    beta_increment_per_sampling = 0.001
    _max_priority = 1.0

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity=capacity

    def _get_priority(self, value):
        return (value + self.e) ** self.a
    
    def add(self, data):
        # priority = self._get_priority(value)
        self.tree.add(self._max_priority ** self.a, data)

    def sample(self, batch_size):

        '''
        Returns obs, acts, returns, idxs, is_weights
        '''
        observations, actions, returns = [], [], []
        idxs = []
        segment = self.tree.total()/batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            
            a = segment * i
            b = segment * (i + 1)

            sum = random.uniform(a,b)

            (idx, priority, data) = self.tree.get(sum)

            obs, action, R = data

            priorities.append(priority)
            
            observations.append(np.array(obs))
            actions.append(np.array(action))
            returns.append(R)

            idxs.append(idx)

            sampling_probabilities = priorities / self.tree.total()
            is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
            is_weight /= is_weight.max()

        return np.array(observations), np.array(actions), np.array(returns), idxs, is_weight
        
    def update_priority(self, idx, value):
        
        priority = self._get_priority(value)
        self.tree.update(idx,priority)

    
    def update_priorities(self, idxs, priorities):

        for idx, priority in zip(idxs, priorities):

            priority = max(priority, 1e-6)

            assert priority > 0
            self.tree.update(idx, priority ** self.a)