import tensorflow as tf
import gym
from gym import wrappers
import numpy as np
import time
import os
import datetime

from tensorflow.contrib import layers
from tensorflow.core.framework import summary_pb2

GAMMA = 0.99

# Entropy weight.
BETA = 0.01


def build_shared_policy_and_value(input_state, num_actions, reuse=False):
    with tf.variable_scope("torso", reuse=reuse):
        out = layers.flatten(input_state)
        out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
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


def main():
    # gym.upload("/tmp/cartpole-experiment-1", api_key="sk_scDpKlQS7ercGiVUZWEgw")
    # return
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    # env = wrappers.Monitor(env, "/tmp/cartpole-experiment-1")

    seed = 42
    env.seed(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)

    episode_count = 1000000
    max_steps = 10000

    num_actions = env.action_space.n
    input_state_shape = list(env.observation_space.shape)
    input_state_ph = tf.placeholder(dtype=tf.float32, shape=[None] + input_state_shape, name="input")
    next_input_state_ph = tf.placeholder(dtype=tf.float32, shape=[None] + input_state_shape, name="next_input")

    # builder = build_separate_policy_and_value
    builder = build_shared_policy_and_value
    policy, value_function = builder(input_state_ph, num_actions)
    _, next_value_function = builder(next_input_state_ph, num_actions, reuse=True)

    def select(data, indices):
        return tf.reduce_sum(data * tf.one_hot(indices, 2, axis=-1), axis=1)

    action_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="action_taken")
    rewards_ph = tf.placeholder(dtype=tf.float32, shape=[None], name="rewards")
    is_terminal_ph = tf.placeholder(dtype=tf.float32, shape=[None], name="is_terminal_ph")
    advantage = rewards_ph + GAMMA * tf.stop_gradient(next_value_function) * (1.0 - is_terminal_ph) - value_function
    tf.summary.histogram("value_function", value_function)
    tf.summary.histogram("advantage", advantage)
    with tf.variable_scope("action_log_policy"):
        action_log_policy = select(tf.log(policy), action_ph)

    entropy = -tf.reduce_mean(tf.reduce_sum(policy * tf.log(policy), axis=1))
    tf.summary.scalar("entropy", entropy)

    trainable_vars = tf.trainable_variables()
    policy_vars = trainable_vars
    value_vars = trainable_vars

    policy_optimizer = tf.train.AdamOptimizer()
    value_optimizer = tf.train.RMSPropOptimizer(0.00025)

    with tf.variable_scope("loss"):
        policy_loss = -tf.reduce_mean(tf.stop_gradient(advantage) * action_log_policy, name="policy_loss")
        policy_loss_with_entropy = policy_loss - BETA * entropy
        tf.summary.scalar("policy_loss", policy_loss)
        value_loss = tf.reduce_mean(tf.square(advantage), name="value_loss")
        tf.summary.scalar("value_loss", value_loss)
        update_vars = tf.group(policy_optimizer.minimize(policy_loss), value_optimizer.minimize(value_loss))

    summaries_dir = "/tmp/cartpole/"
    merged = tf.summary.merge_all()

    tf_config = tf.ConfigProto(log_device_placement=False)

    batch_size = 5000
    simulation_depth = 5

    with tf.train.SingularMonitoredSession(config=tf_config) as session:
        summary_name = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
        summary_writer = tf.summary.FileWriter(
                os.path.join(summaries_dir, summary_name), session.graph)

        start_time = time.clock()
        total_frames = 0

        states = []
        next_states = []
        actions = []
        rewards = []
        is_terminal_values = []

        def train_on_batch():
            nonlocal states
            nonlocal next_states
            nonlocal actions
            nonlocal rewards
            nonlocal is_terminal_values

            summary, _ = session.run([merged, update_vars], feed_dict={
                input_state_ph: states,
                next_input_state_ph: next_states,
                action_ph: actions,
                rewards_ph: rewards,
                is_terminal_ph: is_terminal_values,
                })

            summary_writer.add_summary(summary, episode)

            states = []
            next_states = []
            actions = []
            rewards = []
            is_terminal_values = []

        for episode in range(episode_count):
            state = env.reset()


            total_reward = 0
            total_rewards = []

            show_episode = episode % 100 == 0

            for step in range(max_steps):
                if show_episode:
                    env.render()

                probs = policy.eval(session=session, feed_dict={input_state_ph: [state]})[0]
                action = np.random.choice(num_actions, p=probs)

                next_state, reward, is_terminal, info = env.step(action)
                states.append(state)
                next_states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                is_terminal_values.append(is_terminal)

                total_reward += reward

                # if len(states) >= batch_size:
                    # train_on_batch()

                if is_terminal:
                    train_on_batch()

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
