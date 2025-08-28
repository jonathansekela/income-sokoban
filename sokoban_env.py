# @author Jonathan Sekela

from typing import Optional
import numpy as np
import gymnasium as gym

"""
make it a 2d array of ints. put wall around all edges.
0: w = wall
1: g = green floor
2: b = blue floor
3: p = purple floor
4: b = black floor
5: o = orange floor
"""

class SokobanEnv(gym.Env):

# region constructors

	# @todo: reimplement given new location and movement stuff
	def __init__(self, max_actions: int = 100, size: int = 10):
		super().__init__()
		# currently a 10x10 board. green room north, black room south, 2-space way between. surrounded by walls.
		self.size = size
		self.board = np.array([	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
								[0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
								[0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
								[0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
								[0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
								[0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
								[0, 2, 2, 2, 2, 2, 2, 2, 2, 0],
								[0, 2, 2, 2, 2, 2, 2, 2, 2, 0],
								[0, 2, 2, 2, 2, 2, 2, 2, 2, 0],
								[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
		
		self.agent_loc = np.array([1, 1]) # x, y coordinate
		self.target_loc = np.array([7, 7]) # x, y coordinate
		
		self.max_actions = max_actions # max number of actions allowed in a single game
		self.actions_taken = 0

		# Define what actions are available (4 directions)
		self.action_space = gym.spaces.Discrete(4)

		# Define what the agent can observe
		# Dict space gives us structured, human-readable observations
		self.observation_space = gym.spaces.Dict(
			{
				"agent": gym.spaces.Box(-1, 1, shape=(2,), dtype=int),   # [x, y] coordinates
				"target": gym.spaces.Box(-1, 1, shape=(2,), dtype=int),  # [x, y] coordinates
				"board": gym.spaces.Box(0, 5, shape=(size, size), dtype=int) # the board: 2d array size x size
			}
		)

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
	## 	seed: Random seed for reproducible episodes
	## 	options: Additional configuration (unused in this example)
	## Returns tuple: (observation, info) for the initial state
	def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
		# IMPORTANT: Must call this first to seed the random number generator
		super().reset(seed=seed)
		self.actions_taken = 0

		# @todo: randomly place the agent anywhere on the grid
		self.agent_loc = np.array([1, 1])
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
	def step(self, action):

		self.actions_taken += 1
		# Map the discrete action (0-3) to a movement direction
		direction = self._action_to_direction[action.item()] # @todo: when model doesn't use numpy.int64, change this back to just action

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
		return self.board[test[0]][test[1]] == 0
	
	# goal reached if target is in green room
	def __goal_reached(self):
		return self.board[self.target_loc[0]][self.target_loc[1]] == 1
# endregion
# endregion