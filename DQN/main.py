#!/usr/bin/env python
import logging
import os
import sys
sys.path.append("games/")
import pong_fun as game # whichever is imported "as game" will be used

import tensorflow as tf
import gym

from agent_trainer import AgentTrainer

# Model
GAME = "pong"
ACTIONS = 3  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 10000.  # timesteps to observe before training
EXPLORE = 2000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 1  # starting value of epsilon
REPLAY_MEMORY = 100000  # number of previous transitions to remember
MATCH_MEMORY = 1000  # number of previous matches to remember
BATCH_SIZE = 64  # size of minibatch
FRAME_PER_ACTION = 1  # ammount of frames that are skipped before every action
MODEL_PATH = "./saved_networks"  # path to saved models
SNAPSHOT_PERIOD = 10000  # periodicity of saving current model

# Training
NUM_THREADS = 3  # number of threads for tensorflow session

# Logging
LOG_PERIOD = 100  # periodicity of logging
LOG_PATH = "./logs"  # path to logs
LOG_FILE = os.path.join(LOG_PATH, "tf.log")  # path to logs
LOG_TIMINGS = False  # Whether to log controller speed on every tick
tf.logging.set_verbosity(tf.logging.DEBUG)

handler = logging.FileHandler(LOG_FILE)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
logging.getLogger('tensorflow').addHandler(handler)

config = {
    "num_threads": 3,
    "action_count": ACTIONS,
    "gamma": 0.99,
    "observe_step_count": 10000,
    "explore_step_count": 2000000,
    "initial_epsilon": 1.0,
    "final_epsilon": 0.0001,
    "replay_memory_size": 100000,
    "match_memory_size": 1000,
    "batch_size": 64,
    "frame_per_action": 1,
    "log_period": 100,
}


def vectorFromAction(action_index):
    import numpy as np
    a = np.zeros(ACTIONS)
    a[action_index] = 1
    return a


def playGame():
    # Open up a game state to communicate with emulator
    # env = gym.make("Pong-v0")
    env = game.GameState()

    trainer = AgentTrainer(config)
    trainer.init_training()
    trainer.load_model(MODEL_PATH)

    episode_count = 1000000
    max_steps = 1000000

    step_count = 0
    for episode in range(episode_count):
        # get the first state by doing nothing and preprocess the image to 80x80x4
        # x_t = env.reset()
        x_t, _, _ = env.frame_step(vectorFromAction(0))
        trainer.reset_state(x_t)

        for step in range(max_steps):
            # env.render()
            # choose an action epsilon greedily
            action_index = trainer.act()

            # run the selected action and observe next state and reward
            # x_t1, r_t, terminal, _ = env.step(action_index)
            x_t1, r_t, terminal = env.frame_step(vectorFromAction(action_index))
            trainer.process_frame(x_t1, r_t, terminal)

            step_count += 1
            if step_count % SNAPSHOT_PERIOD == 0:
                trainer.save_model(MODEL_PATH)

            if terminal:
                break


def main():
    playGame()


if __name__ == "__main__":
    main()
