from MineralEnv import MineralEnv

from absl import flags

import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts


FLAGS = flags.FLAGS
FLAGS([''])

python_environment = MineralEnv()
tf_env = tf_py_environment.TFPyEnvironment(python_environment)

time_step = tf_env.reset()
rewards = []
steps = []
number_of_episodes = 10000

for _ in range(number_of_episodes):
    reward_t = 0
    steps_t = 0
    tf_env.reset()
    while True:
        action = tf.random.uniform([1], 0, 9, dtype=tf.int32)
        next_time_step = tf_env.step(action)
        if tf_env.current_time_step().is_last():
            break
        steps_t += 1
        reward_t += next_time_step.reward.numpy()
    rewards.append(reward_t)
    steps.append(steps_t)

print(rewards)
print(steps)