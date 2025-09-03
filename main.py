# @author Jonathan Sekela

# @todo: setup eval pipeline for each agent in main. env is deterministic. cumulative reward over an episode. train for a number of episodes, freeze policy, test on that policy for a number of episodes, average cumulative reward over all episodes, plot that point, repeat.
# keep in mind the policy is an object, so when you test the network you're just querying the policy object.

# general packages
import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C
# custom packages
import sokoban_env
import agent

env = sokoban_env.SokobanEnv(max_actions = 200)

model = A2C("MultiInputPolicy", env, verbose=1, device='cuda')
model.learn(total_timesteps=3000)
model.save("a2c_sokoban")

del model # remove to demonstrate saving and loading

model = A2C.load("a2c_sokoban")

# run a set number of episodes
terminated_eps = 0
truncated_eps = 0
for episode in range(100):
	observation, info = env.reset()
	terminated = False
	truncated = False
	while not (terminated or truncated):
		# this is where you would insert your policy. currently random.
		action, _states = model.predict(observation)
		# action = env.action_space.sample()

		# step (transition) through the environment with the action
		# receiving the next observation, reward and if the episode has terminated or truncated
		observation, reward, terminated, truncated, info = env.step(action)
		# print(observation)

		# If the episode has ended then we can reset to start a new episode
		if terminated or truncated:
			if terminated:
				terminated_eps += 1
			else:
				truncated_eps += 1
			print("%s: %s" % (episode + 1, 'TERMINATED' if terminated else 'TRUNCATED'))
			observation, info = env.reset()
print("terminated total: %s | truncated total: %s" % (terminated_eps, truncated_eps))
env.close()