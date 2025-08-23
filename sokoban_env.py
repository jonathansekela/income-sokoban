# @author Jonathan Sekela

from typing import Optional
import numpy as np
import gymnasium as gym

"""
make it a 2d array of chars. put wall around all edges.
g = green floor
b = blue floor
p = purple floor
b = black floor
o = orange floor
w = wall
a = agent
c = chair
"""

class SokobanEnv(gym.Env):

# region constructors

	# @todo: reimplement given new location and movement stuff
	def __init__(self, max_actions: int = 100, size: int = 10):
		# currently a 10x10 board. green room north, black room south, 2-space way between. surrounded by walls.
		self.size = size
		self.board = np.array([	['w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w'],
								['w', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'w'],
								['w', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'w'],
								['w', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'w'],
								['w', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'w'],
								['w', 'w', 'w', 'w', 'b', 'b', 'w', 'w', 'w', 'w'],
								['w', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'w'],
								['w', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'w'],
								['w', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'w'],
								['w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w']])
		
		self.agent_loc = np.array([1, 1]) # x, y coordinate
		self.target_loc = np.array([7, 7]) # x, y coordinate
		
		self.max_actions = max_actions # max number of actions allowed in a single game
		self.actions_taken = 0

		# Define what actions are available (4 directions)
		self.action_space = gym.spaces.Discrete(4)

		# Map action numbers to actual movements on the grid
		# This makes the code more readable than using raw numbers
		self._action_to_direction = {
			0: np.array([1, 0]),   # Move right (positive x)
			1: np.array([0, 1]),   # Move up (positive y)
			2: np.array([-1, 0]),  # Move left (negative x)
			3: np.array([0, -1]),  # Move down (negative y)
		}
# endregion

# region public methods

	## Start a new episode.
	## Args:
	## 	seed: Random seed for reproducible episodes
	## 	options: Additional configuration (unused in this example)
	## Returns:
	## 	tuple: (observation, info) for the initial state
	# @todo: reimplement
	def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
		# IMPORTANT: Must call this first to seed the random number generator
		super().reset(seed=seed)

		# @todo: randomly place the agent anywhere on the grid
		self.agent_loc = np.array([2, 2])
		# self.agent_loc = self.np_random.integers(0, self.size, size=2, dtype=int)

		# @todo: randomly place target, ensuring it's different from agent position
		self.target_loc = np.array([7, 7])
		# self.target_loc = self.agent_loc
		# while np.array_equal(self.target_loc, self.agent_loc):
		# 	self.target_loc = self.np_random.integers(
		# 		0, self.size, size=2, dtype=int
		# 	)

		observation = self._get_obs()
		info = self._get_info()

		return observation, info

	## Execute one timestep within the environment.
	## Args:
	## 	action: The action to take (0-3 for directions)
	## Returns:
	## 	tuple: (observation, reward, terminated, truncated, info)
	# @todo: reimplement
	def step(self, action):

		self.actions_taken += 1
		# Map the discrete action (0-3) to a movement direction
		direction = self._action_to_direction[action]

		# Update agent position, ensuring it stays within grid bounds
		# np.clip prevents the agent from walking off the edge
		if not self.__is_wall(self.agent_loc, direction):
			# if agent moves into chair, check that chair won't move into wall then move both in direction
			if np.array_equal(self.agent_loc + direction, self.target_loc) and not self.__is_wall(self.target_loc, direction):
				self.agent_loc = np.clip(self.agent_loc + direction, 0, self.size - 1)
				self.target_loc = np.clip(self.target_loc + direction, 0, self.size - 1)
			# if not a wall space, move agent in direction
			else:
				self.agent_loc = np.clip(self.agent_loc + direction, 0, self.size - 1)

		# Check if agent reached the target
		terminated = self.__goal_reached()
		# terminated = np.array_equal(self.agent_loc, self.target_loc)

		# truncate if agent runs out of actions
		truncated = self.actions_taken >= self.max_actions

		# Simple reward structure: +1 for reaching target, 0 otherwise
		# Alternative: could give small negative rewards for each step to encourage efficiency
		reward = 1 if terminated else 0

		observation = self._get_obs()
		info = self._get_info()

		return observation, reward, terminated, truncated, info
# endregion
	
# region private methods

	## Convert internal state to observation format.
	## Returns:
	## 	dict: Observation with agent and target positions
	# @todo: reimplement
	def _get_obs(self):
		return {"agent": self.agent_loc, "target": self.target_loc, "board": self.board}
	
	"""Compute auxiliary information for debugging.
	Returns:
		dict: Info with distance between agent and target
	"""
	# @todo: reimplement
	def _get_info(self):
		return {
			"distance": np.linalg.norm(
				self.agent_loc - self.target_loc, ord=1
			)
		}

# region utility methods

	def __is_wall(self, location, direction):
		test = location + direction
		return self.board[test[0]][test[1]] == 'w'
	
	# goal reached if target is in green room
	def __goal_reached(self):
		return self.board[self.target_loc[0]][self.target_loc[1]] == 'g'
# endregion
# endregion