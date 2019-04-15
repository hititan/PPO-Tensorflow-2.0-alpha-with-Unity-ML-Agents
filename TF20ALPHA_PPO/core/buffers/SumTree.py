import numpy as np

'''
Sum Tree is binary tree data structure where the parents value is the sum of its children priorities. 
In the leaf nodes you can find priorities and the corresponding data-object (o,a,r)

Goal is searching fast for a priority which is sampled by a random uniform distribution
Priority is searched over SUM of priorities

So you dont have to go through a sorted Array from max priority to min priority 
Probability of being sampled is Pi = priority[i] / Sum od all priorities
'''

class SumTree:

    data_idx = 0

    def __init__(self, capacity):

        self.capacity = capacity
        self.tree = np.zeros(2*capacity-1) # Because of Binary structure 
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        
    def _propagate(self,idx,change):

        '''
        updates everything from idx as start to the root node
        '''
        parent = (idx-1) // 2

        self.tree[parent] +=change

        # Recursion until root node is reached
        if parent !=0:
            self._propagate(parent,change)

    def _retrieve(self, idx, sum):
        '''
        Find a sample on a leaf node; Left nodes are less than the root nodes which is less than the right nodes

                    N
                   / \
                 C1   C2

                 In such a Node trinagle values always ar getting bigger from left to the right  C1 < N < C2

                 Searching for a Value:

                 if Value is bigger on Node go to the right side 
                 if Value is smaller on Node go to the left side
        '''

        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx
        
        if sum <= self.tree[left]:
            return self._retrieve(left, sum)

        else: 
            return self._retrieve(right, sum - self.tree[left])

        
    def total(self):
        return self.tree[0]
    
    def add(self, priority, data):
        '''
        Stores a priority and the Date belonging to the priority
        '''
        idx = self.data_idx + self.capacity - 1

        self.data[self.data_idx] = data
        self.update(idx, priority)


        self.data_idx +=1

        if self.data_idx >= self.capacity:
            self.data_idx = 0
        
        if self.n_entries < self.capacity:
            self.n_entries +=1

    def update(self, idx, priority):
        '''
        Update Priority in Tree
        '''

        change = priority - self.tree[idx]

        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, sum):
        '''
        Get Priority and Sample
        '''
        idx = self._retrieve(0,sum)
        data_idx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[data_idx])
