from MineralEnv import MineralEnv

from absl import flags

import tensorflow as tf
import numpy as np

import reverb
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils

import tf_agents
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import dynamic_step_driver
from tf_agents.policies import random_tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.drivers import py_driver
from tf_agents.policies import py_tf_eager_policy

from dqn_cnn import build_model
from util import compute_avg_return

# Flags needed for creating pysc2 environment
FLAGS = flags.FLAGS
FLAGS([''])

# Hyperparameters
num_eval_episodes = 1
initial_collect_steps = 10
collect_steps_per_iteration = 1
replay_buffer_max_length = 500
log_interval = 1
eval_interval = 1

num_iterations = 2

batch_size = 64
lr = 0.0005

# Create CollectMineralShards python environment
python_environment = MineralEnv()
# Cast python environment to tensorflow environment
sc2_env = tf_py_environment.TFPyEnvironment(python_environment)
sc2_eval_env = tf_py_environment.TFPyEnvironment(python_environment)

# Build network model for DQN
q_net = build_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
train_step_counter = tf.Variable(0)
# Create DQN
agent = dqn_agent.DqnAgent(
    sc2_env.time_step_spec(),
    sc2_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter
)
agent.initialize()
policy = agent.policy
collect_policy = agent.collect_policy

random_policy = random_tf_policy.RandomTFPolicy(sc2_env.time_step_spec(),
                                                sc2_env.action_spec())
time_step = sc2_env.reset()                                                
#print(random_policy.action(time_step))
#compute_avg_return(sc2_eval_env, random_policy, num_eval_episodes)

table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(
      agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)

table = reverb.Table(
    table_name,
    max_size=replay_buffer_max_length,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    agent.collect_data_spec,
    table_name=table_name,
    sequence_length=2,
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  replay_buffer.py_client,
  table_name,
  sequence_length=2)
print("@@@")
# py_driver.PyDriver(
#     sc2_env,
#     py_tf_eager_policy.PyTFEagerPolicy(
#       random_policy, use_tf_function=True),
#     [rb_observer],
#     max_steps=initial_collect_steps).run(sc2_env.reset())

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

iterator = iter(dataset)

# Reset the train step.
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(sc2_eval_env, agent.policy, 1)
returns = [avg_return]
print("$$$")
# Reset the environment.
time_step = sc2_env.reset()

# Create a driver to collect experience.
collect_driver = py_driver.PyDriver(
    sc2_env,
    py_tf_eager_policy.PyTFEagerPolicy(
      agent.collect_policy, use_tf_function=True),
    [rb_observer],
    max_steps=collect_steps_per_iteration)
print("***")
for eps in range(num_iterations):
    print("Episode", eps)
    # Collect a few steps and save to the replay buffer.
    time_step, _ = collect_driver.run(time_step)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(sc2_eval_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)

print("Done")
sc2_env.pyenv.envs[0].env.close()
exit()

print("@@@ Training Finished. @@@")
# Testing below
time_step = sc2_env.reset()
rewards = []
steps = []
number_of_episodes = 10000

for _ in range(number_of_episodes):
    reward_t = 0
    steps_t = 0
    sc2_env.reset()
    while True:
        action = tf.random.uniform([1], 0, 9, dtype=tf.int32)
        next_time_step = sc2_env.step(action)
        if sc2_env.current_time_step().is_last():
            break
        steps_t += 1
        reward_t += next_time_step.reward.numpy()
    rewards.append(reward_t)
    steps.append(steps_t)

print(rewards)
print(steps)