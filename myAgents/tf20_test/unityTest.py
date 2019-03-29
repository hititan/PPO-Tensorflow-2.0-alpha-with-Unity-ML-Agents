from __future__ import absolute_import, division, print_function
import logging
import tensorflow as tf
import numpy as np

from mlagents.envs import UnityEnvironment
#print(tf.version)


env = UnityEnvironment(file_name=None, worker_id=0, seed=1)

print(str(env))

info = env.reset()

brain = info['RobotBrain']
print(brain.vector_observations)

env.close()

