#!/usr/bin/env python
from agent_trainer import AgentTrainer

import tensorflow as tf
import numpy as np
import gym


# Experiment description.
GAME = "gym_pong"
MODEL = "dqn"
VERSION = 1  # Bump this for each new experiment.

EXPERIMENT_PATH = os.path.join("/home/acid", GAME, MODEL, str(VERSION))
# "/home/acid/Repos/HSE_AI_Labs/Lab_4/saved_networks"  # path to saved models
MODEL_PATH = os.path.join(EXPERIMENT_PATH, "checkpoints")
LOG_PATH = os.path.join(EXPERIMENT_PATH, "logs")

# Model
SNAPSHOT_PERIOD = 10000  # periodicity of saving current model
SEED = 42


def createGameConfig(env):
    return {
        "action_count": env.action_space.n,  # number of valid actions
        "gamma": 1.0,  # decay rate of past observations
        "observe_step_count": 10000,  # timesteps to observe before training
        "explore_step_count": 2000000,  # frames over which to anneal epsilon
        "initial_epsilon": 1.0,  # initial value of epsilon
        "final_epsilon": 0.0001,  # final value of epsilon
        "replay_memory_size": 100000,  # number of previous transitions to remember
        "match_memory_size": 1000,  # number of previous matches to remember
        "batch_size": 256,  # size of minibatch
        "frame_per_action": 1,  # ammount of frames that are skipped before every action
        "log_period": 100,  # periodicity of logging
        "experiment_path": EXPERIMENT_PATH,
    }


def playGame(game_name):
    np.random.seed(SEED)
    tf.set_random_seed(SEED)

    # Open up a game state to communicate with emulator
    if game_name == "PyGamePong-v0":
        from games.pong import PongEnv
        env = PongEnv()
    else:
        env = gym.make(game_name)

    trainer = AgentTrainer(createGameConfig(env))
    trainer.init_training()
    trainer.load_model(MODEL_PATH)

    episode_count = 1000000
    max_steps = 1000000

    step_count = 0
    for episode in range(episode_count):
        # get the first state by doing nothing and preprocess the image to 80x80x4
        x_t = env.reset()
        trainer.reset_state(x_t)

        for step in range(max_steps):
            env.render()

            action_index = trainer.act()

            # run the selected action and observe next state and reward
            x_t1, r_t, terminal, _ = env.step(action_index)
            trainer.process_frame(x_t1, r_t, terminal)

            step_count += 1
            if step_count % SNAPSHOT_PERIOD == 0:
                trainer.save_model(MODEL_PATH)

            if terminal:
                break


def main():
    playGame("PyGamePong-v0")


if __name__ == "__main__":
    main()
