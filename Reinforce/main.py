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


def build_policy_and_value(input_state, num_actions):
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

    def get_policy_step_size(t):
        base_step_size = 0.001
        return base_step_size / ((t + 1)**0.6)

    def get_value_step_size(t):
        base_step_size = 0.001
        return base_step_size / ((t + 1)**0.6)

    num_actions = env.action_space.n
    input_state_shape = list(env.observation_space.shape)
    input_state_ph = tf.placeholder(dtype=tf.float32, shape=[None] + input_state_shape, name="input")
    policy, value_function = build_policy_and_value(input_state_ph, num_actions)

    def select(data, indices):
        return tf.reduce_sum(data * tf.one_hot(indices, 2, axis=-1), axis=1)

    policy_step_size_ph = tf.placeholder(dtype=tf.float32, shape=[], name="policy_step_size")
    value_step_size_ph = tf.placeholder(dtype=tf.float32, shape=[], name="value_step_size")
    action_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="action_taken")
    discounted_return_ph = tf.placeholder(dtype=tf.float32, shape=[None], name="discounted_return")
    gamma_power_ph = tf.placeholder(dtype=tf.float32, shape=[None], name="gamma_powers")
    advantage = tf.stop_gradient(discounted_return_ph - value_function, name="advantage")
    tf.summary.histogram("value_function", value_function)
    with tf.variable_scope("action_log_policy"):
        action_log_policy = select(tf.log(policy), action_ph)

    trainable_vars = tf.trainable_variables()
    policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "policy")
    value_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "value")

    policy_gradients = tf.gradients(tf.reduce_sum(policy_step_size_ph * gamma_power_ph * advantage * action_log_policy), policy_vars,
            name="policy_gradients")
    value_gradients = tf.gradients(tf.reduce_sum(value_step_size_ph * advantage * value_function), value_vars,
            name="value_gradients")

    tf.summary.scalar("step_size", policy_step_size_ph)

    with tf.variable_scope("apply_gradients"):
        with tf.variable_scope("update_policy"):
            update_vars_ops = []
            for var, grad in zip(policy_vars, policy_gradients):
                if grad is not None:
                    update_vars_ops.append(tf.assign_add(var, grad))

        with tf.variable_scope("update_value"):
            for var, grad in zip(value_vars, value_gradients):
                if grad is not None:
                    update_vars_ops.append(tf.assign_add(var, tf.clip_by_norm(grad, 1)))

        update_vars = tf.group(*update_vars_ops)

    summaries_dir = "/tmp/cartpole/"
    merged = tf.summary.merge_all()


    with tf.train.SingularMonitoredSession() as session:
        summary_name = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
        summary_writer = tf.summary.FileWriter(
                os.path.join(summaries_dir, summary_name), session.graph)

        start_time = time.clock()
        total_frames = 0
        for episode in range(episode_count):
            state = env.reset()

            states = []
            actions = []
            rewards = []

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
                actions.append(action)
                rewards.append(reward)

                total_reward += reward

                if is_terminal:
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

                        states = np.array(states)
                    actions = np.array(actions)
                    discounted_returns = np.array(rewards)
                    gamma_powers = np.ones(len(rewards))
                    for i in range(1, len(gamma_powers)):
                        gamma_powers[i] = GAMMA * gamma_powers[i - 1]
                    for i in range(len(discounted_returns) - 2, 0, -1):
                        discounted_returns[i] += GAMMA * discounted_returns[i + 1]

                    summary, _ = session.run([merged, update_vars], feed_dict={
                        input_state_ph: states,
                        action_ph: actions,
                        discounted_return_ph: discounted_returns,
                        gamma_power_ph: gamma_powers,
                        policy_step_size_ph: get_policy_step_size(episode),
                        value_step_size_ph: get_value_step_size(episode),
                        })

                    summary_writer.add_summary(summary, episode)
                    break
                else:
                    state = next_state

if __name__ == "__main__":
    main()
