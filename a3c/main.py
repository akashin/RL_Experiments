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

GAMMA = 0.99

# Entropy weight.
BETA = 0.01

# Max length of the episode.
MAX_LENGTH = 500

class ValueFunctionVisualizer(object):

    def __init__(self, window_size):
        self.window_size = window_size
        self.xdata = range(window_size)
        self.ydata = [0 for _ in range(window_size)]

        self.fig, self.ax = plt.subplots(figsize=(12, 12), dpi=80)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(0, self.window_size)
        self.ax.set_ylim(0, 500)
        self.ax.hold(True)
        self.line, = self.ax.plot(self.xdata, self.ydata, 'r-')

        self.tick = 0

        plt.show(block=False)

        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
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
 

def build_shared_policy_and_value(input_state, num_actions, reuse=False):
    with tf.variable_scope("torso", reuse=reuse):
        out = layers.flatten(input_state)
        # out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.relu)
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
    return list(state) + [(MAX_LENGTH - ts) * 1.0 / MAX_LENGTH]


class Actor(object):

    def __init__(self, env, model_builder):
        self.num_actions = env.action_space.n
        input_state_shape = list(env.observation_space.shape)
        # Adding a timestep.
        input_state_shape[0] += 1
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
        input_state_shape[0] += 1
        self.input_state_ph = tf.placeholder(dtype=tf.float32, shape=[None] + input_state_shape, name="input")
        self.next_input_state_ph = tf.placeholder(dtype=tf.float32, shape=[None] + input_state_shape, name="next_input")

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

        value_learning_rate = 1e-2

        policy_optimizer = tf.train.AdamOptimizer()
        value_optimizer = tf.train.RMSPropOptimizer(value_learning_rate)

        with tf.variable_scope("loss"):
            policy_loss = -tf.reduce_mean(tf.stop_gradient(advantage) * action_log_policy, name="policy_loss")
            policy_loss_with_entropy = policy_loss - BETA * entropy
            tf.summary.scalar("policy_loss", policy_loss)
            value_loss = tf.reduce_mean(tf.square(advantage), name="value_loss")
            tf.summary.scalar("value_loss", value_loss)
            self.train_op = tf.group(policy_optimizer.minimize(policy_loss), value_optimizer.minimize(value_loss))

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


def main():
    # gym.upload("/tmp/cartpole-experiment-1", api_key="sk_scDpKlQS7ercGiVUZWEgw")
    # return
    env_name = "CartPole-v1"
    # env_name = "Pong-ram-v0"
    env = gym.make(env_name)
    # env = wrappers.Monitor(env, "/tmp/cartpole-experiment-1")

    seed = 42
    env.seed(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)

    episode_count = 1000000
    max_steps = 10000

    summaries_dir = "/tmp/cartpole/"

    tf_config = tf.ConfigProto(log_device_placement=False)

    num_actions = env.action_space.n
    builder = build_shared_policy_and_value

    batch_size = 128
    simulation_depth = 20

    learner = Learner(env, builder, simulation_depth)
    actor = Actor(env, builder)

    merged_summaries = tf.summary.merge_all()

    value_function_visualizer = ValueFunctionVisualizer(500)

    with tf.train.SingularMonitoredSession(config=tf_config) as session:
        summary_name = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
        summary_writer = tf.summary.FileWriter(
                os.path.join(summaries_dir, summary_name), session.graph)

        start_time = time.clock()
        total_frames = 0

        for episode in range(episode_count):
            state = add_timestep(env.reset(), 0)

            # Episode data.
            states = []
            rewards = []
            actions = []

            total_reward = 0
            total_rewards = []

            show_episode = episode % 100 == 0

            for step in range(max_steps):
                probs, value = actor.eval_actions(session, state)
                action = np.random.choice(num_actions, p=probs)

                if show_episode:
                    env.render()
                    # print(value)
                    value_function_visualizer.add_value(step, value)

                next_state, reward, is_terminal, info = env.step(action)
                next_state = add_timestep(next_state, step + 1)
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                total_reward += reward

                if learner.sample_count() > batch_size:
                    learner.add_samples(states + [next_state], rewards, actions, is_terminal)
                    states = []
                    rewards = []
                    actions = []
                    summary = learner.train(session, merged_summaries)
                    summary_writer.add_summary(summary, episode)

                if is_terminal:
                    learner.add_samples(states + [next_state], rewards, actions, is_terminal)
                    summary = learner.train(session, merged_summaries)
                    summary_writer.add_summary(summary, episode)

                    total_frames += step
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
