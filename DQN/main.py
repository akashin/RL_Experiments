#!/usr/bin/env python
import logging
import os
import sys

import tensorflow as tf
import numpy as np

from agent_trainer import AgentTrainer

import gym
sys.path.append("games/")
import pong_fun as game # whichever is imported "as game" will be used

# Model
GAME = "pong"
ACTIONS = 3  # number of valid actions
MODEL_PATH = "./saved_networks"  # path to saved models
SNAPSHOT_PERIOD = 10000  # periodicity of saving current model
SEED = 42

# Logging
LOG_PATH = "./logs"  # path to logs

config = {
    "action_count": ACTIONS,
    "gamma": 0.99,  # decay rate of past observations
    "observe_step_count": 10000,  # timesteps to observe before training
    "explore_step_count": 2000000,  # frames over which to anneal epsilon
    "initial_epsilon": 1.0,  # starting value of epsilon
    "final_epsilon": 0.0001,  # final value of epsilon
    "replay_memory_size": 100000,  # number of previous transitions to remember
    "match_memory_size": 1000,  # number of previous matches to remember
    "batch_size": 64,  # size of minibatch
    "frame_per_action": 1,  # ammount of frames that are skipped before every action
    "log_period": 100,  # periodicity of logging
}


def vectorFromAction(action_index):
    a = np.zeros(ACTIONS)
    a[action_index] = 1
    return a


def playGame():
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
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
