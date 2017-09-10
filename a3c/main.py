import tensorflow as tf
import gym
from gym import wrappers
import numpy as np
import time
import os
import datetime

from tensorflow.contrib import layers
from tensorflow.core.framework import summary_pb2

import matplotlib
# matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from atari_wrappers import *

GAMMA = 0.99

# Entropy weight.
BETA = 0.01

# Max length of the episode.
MAX_LENGTH = 500

USE_HUBER = False
DELTA = 10.0

# TASK_NAME="cartpole"
# TASK_NAME="pong"
TASK_NAME="pong-2d"
# TASK_NAME="breakout"


if TASK_NAME == "cartpole":
    ADD_TIMESTAMP = True
else:
    ADD_TIMESTAMP = False

def huber_loss(tensor):
    abs_error = tf.abs(tensor)
    quadratic = tf.minimum(abs_error, DELTA)
    linear = (abs_error - quadratic)
    return 0.5 * quadratic**2 + DELTA * linear


class ValueFunctionVisualizer(object):

    def __init__(self, window_size):
        self.window_size = window_size
        self.xdata = range(window_size)
        self.ydata = [0 for _ in range(window_size)]

        self.fig, self.ax = plt.subplots(figsize=(12, 12), dpi=80)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(0, self.window_size)
        self.ax.set_ylim(-23, 23)
        self.ax.hold(True)
        self.line, = self.ax.plot(self.xdata, self.ydata, 'r-')

        self.tick = 0

        plt.show(block=False)

        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.init_graph()

    def init_graph(self):
        self.fig.canvas.draw()
        plt.draw()

    def _redraw(self):
        self.tick += 1
        if self.tick % 5 != 1:
            return

        self.line.set_ydata(self.ydata)

        self.fig.canvas.restore_region(self.background)
        self.ax.draw_artist(self.line)
        self.fig.canvas.blit(self.ax.bbox)

    def add_value(self, ts, value):
        self.ydata.append(value)
        self.ydata = self.ydata[-self.window_size:]
        self._redraw()

def build_shared_policy_and_value_2d(input_state, num_actions, reuse=False):
    with tf.variable_scope("torso", reuse=reuse):
        out = input_state
        out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
        out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
        out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        out = layers.fully_connected(out, num_outputs=512, activation_fn=tf.nn.relu)

        with tf.variable_scope("policy"):
            policy_out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
            policy = layers.softmax(policy_out)

        with tf.variable_scope("value"):
            value_out = layers.fully_connected(out, num_outputs=1, activation_fn=None)
            value = tf.reshape(value_out, [-1])

    return policy, value

def build_shared_policy_and_value(input_state, num_actions, reuse=False):
    with tf.variable_scope("torso", reuse=reuse):
        out = layers.flatten(input_state)
        out = layers.fully_connected(out, num_outputs=256, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
        with tf.variable_scope("policy"):
            policy_out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
            policy = layers.softmax(policy_out)

        with tf.variable_scope("value"):
            value_out = layers.fully_connected(out, num_outputs=1, activation_fn=None)
            value = tf.reshape(value_out, [-1])

    return policy, value

def build_shared_policy_and_value_simple(input_state, num_actions, reuse=False):
    with tf.variable_scope("torso", reuse=reuse):
        out = layers.flatten(input_state)
        # out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=256, activation_fn=tf.nn.relu)
        with tf.variable_scope("policy"):
            policy_out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
            policy = layers.softmax(policy_out)

        with tf.variable_scope("value"):
            value_out = layers.fully_connected(out, num_outputs=1, activation_fn=None)
            value = tf.reshape(value_out, [-1])

    return policy, value

def build_separate_policy_and_value(input_state, num_actions, reuse=False):
    with tf.variable_scope("torso", reuse=reuse):
        with tf.variable_scope("policy"):
            out = layers.flatten(input_state)
            out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
            policy = layers.softmax(out)

        with tf.variable_scope("value"):
            out = layers.flatten(input_state)
            out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=1, activation_fn=None)
            value = tf.reshape(out, [-1])

    return policy, value


def scalar_summary(name, value):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=value)])


def add_timestep(state, ts):
    if ADD_TIMESTAMP:
        return list(state) + [ts / MAX_LENGTH]
    else:
        return list(state)


def preprocess_state(state, ts):
    if TASK_NAME == "cartpole":
        return add_timestep(state, ts)
    else:
        return add_timestep(state / 256.0, ts)



class Actor(object):

    def __init__(self, env, model_builder):
        self.num_actions = env.action_space.n
        input_state_shape = list(env.observation_space.shape)
        # Adding a timestep.
        input_state_shape[0] += ADD_TIMESTAMP
        self.input_state_ph = tf.placeholder(dtype=tf.float32, shape=[None] + input_state_shape, name="input")
        self.policy, self.value = model_builder(self.input_state_ph, self.num_actions, reuse=True)

    def eval_actions(self, session, state):
        policy, value = session.run([self.policy, self.value], feed_dict={self.input_state_ph: [state]})
        return policy[0], value[0]


def select(data, indices, num_actions):
    return tf.reduce_sum(data * tf.one_hot(indices, num_actions, axis=-1), axis=1)


class Learner(object):

    def __init__(self, env, model_builder, depth):
        self.reset_state()

        self.depth = depth

        num_actions = env.action_space.n
        input_state_shape = list(env.observation_space.shape)
        input_state_shape[0] += ADD_TIMESTAMP
        self.input_state_ph = tf.placeholder(dtype=tf.float32, shape=[None] + input_state_shape, name="input")
        self.next_input_state_ph = tf.placeholder(dtype=tf.float32, shape=[None] + input_state_shape, name="next_input")

        # tf.summary.image("input", tf.reshape(self.input_state_ph, [-1, 4, 32, 1]))
        # tf.summary.image("input", tf.reshape(self.input_state_ph, [-1, 1, input_state_shape[0], 1]))

        policy, value_function = model_builder(self.input_state_ph, num_actions)
        _, next_value_function = model_builder(self.next_input_state_ph, num_actions, reuse=True)

        # Training data placeholders.
        self.action_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="action_taken")
        self.rewards_ph = tf.placeholder(dtype=tf.float32, shape=[None], name="rewards")
        self.is_terminal_ph = tf.placeholder(dtype=tf.float32, shape=[None], name="is_terminal")

        self.gamma_power_ph = tf.placeholder(dtype=tf.float32, shape=[None], name="gamma_power")

        advantage = self.rewards_ph + self.gamma_power_ph * tf.stop_gradient(next_value_function) * (1.0 - self.is_terminal_ph) - value_function
        tf.summary.histogram("value_function", value_function)
        tf.summary.histogram("advantage", advantage)
        with tf.variable_scope("action_log_policy"):
            action_log_policy = select(tf.log(policy), self.action_ph, num_actions)

        entropy = -tf.reduce_mean(tf.reduce_sum(policy * tf.log(policy), axis=1))
        tf.summary.scalar("entropy", entropy)

        trainable_vars = tf.trainable_variables()
        policy_vars = trainable_vars
        value_vars = trainable_vars

        value_learning_rate = 1e-4

        policy_optimizer = tf.train.AdamOptimizer()
        value_optimizer = tf.train.RMSPropOptimizer(value_learning_rate)

        with tf.variable_scope("loss"):
            policy_loss = -tf.reduce_mean(tf.stop_gradient(advantage) * action_log_policy, name="policy_loss")
            policy_loss_with_entropy = policy_loss - BETA * entropy
            tf.summary.scalar("policy_loss", policy_loss)
            if USE_HUBER:
                value_loss = tf.reduce_mean(huber_loss(advantage), name="value_loss")
            else:
                value_loss = tf.reduce_mean(tf.square(advantage), name="value_loss")
            tf.summary.scalar("value_loss", value_loss)

            def minimize_clipped(optimizer, loss):
                gvs = optimizer.compute_gradients(loss)
                clipped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs if grad is not None]
                # clipped_gvs = gvs
                return optimizer.apply_gradients(clipped_gvs)

            self.train_op = tf.group(minimize_clipped(policy_optimizer, policy_loss),
                                     minimize_clipped(value_optimizer, value_loss))

    def add_samples(self, states, rewards, actions, last_is_terminal):
        N = len(actions)

        for i in range(N):
            self.states_.append(states[i])
            j = min(i + self.depth, N)
            self.next_states_.append(states[j])
            self.gamma_powers_.append(GAMMA ** (j - i))

            r = 0
            for k in range(j - 1, i - 1, -1):
                r = r * GAMMA + rewards[k]
            self.rewards_.append(r)

            self.is_terminal_.append(last_is_terminal * (j == N))

        # self.states_.extend(states[:-1])
        # self.next_states_.extend(states[1:])
        # self.rewards_.extend(rewards)
        self.actions_.extend(actions)

        # self.is_terminal_.extend([0 for _ in range(N)])
        # self.is_terminal_[-1] = last_is_terminal

    def sample_count(self):
        return len(self.states_)

    def reset_state(self):
        self.states_ = []
        self.next_states_ = []
        self.actions_ = []
        self.rewards_ = []
        self.is_terminal_ = []
        self.gamma_powers_ = []

    def train(self, session, summary_op):
        summary, _ = session.run([summary_op, self.train_op], feed_dict={
            self.input_state_ph: self.states_,
            self.next_input_state_ph: self.next_states_,
            self.action_ph: self.actions_,
            self.rewards_ph: self.rewards_,
            self.is_terminal_ph: self.is_terminal_,
            self.gamma_power_ph: self.gamma_powers_,
            })

        self.reset_state()

        return summary


def get_env_by_name(env_name):
    env = gym.make(env_name)
    seed = 42
    env.seed(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    return env


def get_env():
    if TASK_NAME == "pong-2d":
        benchmark = gym.benchmark_spec('Atari40M')
        task = benchmark.tasks[3]
        env_id = task.env_id

        env = get_env_by_name(env_id)
        # env = wrappers.Monitor(env, "/tmp/{}".format(TASK_NAME), force=True)
        env = wrap_deepmind(env)
        return env

    if TASK_NAME == "cartpole":
        env_name = "CartPole-v1"
    elif TASK_NAME == "breakout":
        env_name = "Breakout-ram-v0"
    elif TASK_NAME == "pong":
        env_name = "Pong-ram-v0"

    get_env_by_name(env_name)


def main():
    # gym.upload("/tmp/cartpole-experiment-1", api_key="sk_scDpKlQS7ercGiVUZWEgw")
    # return

    env = get_env()

    episode_count = 1000000
    max_steps = 10000

    summaries_dir = "/tmp/{}".format(TASK_NAME)

    tf_config = tf.ConfigProto(log_device_placement=False)

    num_actions = env.action_space.n
    # builder = build_shared_policy_and_value
    builder = build_shared_policy_and_value_2d

    batch_size = 256
    simulation_depth = 20

    learner = Learner(env, builder, simulation_depth)
    actor = Actor(env, builder)

    saver = tf.train.Saver()
    merged_summaries = tf.summary.merge_all()

    value_function_visualizer = ValueFunctionVisualizer(500)

    SHOW_FREQUENCY = 100
    CHECKPOINT_FREQUENCY = 100

    global_step_tensor = tf.Variable(0, trainable=False, name="global_step")
    tick_global_step = tf.assign_add(global_step_tensor, 1, name="tick_global_step")

    CHECKPOINT_DIR = "/home/acid/RL/{}".format(TASK_NAME)
    saver_hook = tf.train.CheckpointSaverHook(CHECKPOINT_DIR, save_secs=60, saver=saver)

    with tf.train.SingularMonitoredSession(config=tf_config, hooks=[saver_hook]) as session:
        summary_name = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
        summary_writer = tf.summary.FileWriter(
                os.path.join(summaries_dir, summary_name), session.graph)

        checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if checkpoint and checkpoint.model_checkpoint_path:
            print("Restoring from checkpoint {}".format(checkpoint.model_checkpoint_path))
            saver.restore(session, checkpoint.model_checkpoint_path)

        start_time = time.clock()
        total_frames = 0

        for episode in range(episode_count):
            state = preprocess_state(env.reset(), 0)
            session.run(tick_global_step)

            # Episode data.
            states = []
            rewards = []
            actions = []

            total_reward = 0
            total_rewards = []

            show_episode = episode % SHOW_FREQUENCY == 0
            if show_episode:
                value_function_visualizer.init_graph()

            # if episode % CHECKPOINT_FREQUENCY == 0:
                # saver.save(session.raw_session(), 'cartpole-model', global_step=episode)

            for step in range(max_steps):
                probs, value = actor.eval_actions(session, state)
                action = np.random.choice(num_actions, p=probs)

                if show_episode:
                    env.render()
                    value_function_visualizer.add_value(step, value)

                next_state, reward, is_terminal, info = env.step(action)
                next_state = preprocess_state(next_state, step + 1)
                total_frames += 1
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                total_reward += reward

                if len(states) > batch_size:
                    learner.add_samples(states + [next_state], rewards, actions, is_terminal)
                    summary = learner.train(session, merged_summaries)
                    summary_writer.add_summary(summary, total_frames)
                    states = []
                    rewards = []
                    actions = []

                if is_terminal:
                    if rewards:
                        learner.add_samples(states + [next_state], rewards, actions, is_terminal)
                        summary = learner.train(session, merged_summaries)
                        summary_writer.add_summary(summary, total_frames)

                    total_rewards.append(total_reward)
                    if episode % 100 == 0:
                        total_rewards = total_rewards[-100:]
                        avg_reward = np.mean(total_rewards)
                        summary_writer.add_summary(scalar_summary("avg_reward", avg_reward), episode)

                        print("ep: {}, avg: {}, fps: {}".format(
                            episode,
                            avg_reward,
                            total_frames / (time.clock() - start_time)))
                    break
                else:
                    state = next_state

if __name__ == "__main__":
    main()
