# @author Jonathan Sekela

from typing import Optional
import numpy as np
import gymnasium as gym
import sokoban_env
import agent

env = sokoban_env.SokobanEnv()

observation, info = env.reset()
terminated = False
truncated = False

# run a set number of episodes
for _ in range(1):
    while not (terminated or truncated):
        # this is where you would insert your policy. currently random.
        action = env.action_space.sample()

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        observation, reward, terminated, truncated, info = env.step(action)
        print(observation)

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            print("%s" % 'TERMINATED' if terminated else 'TRUNCATED')
            observation, info = env.reset()

env.close()