# -*- coding: utf-8 -*-
from environments.car_controller.car_stuff.alex_discrete.road_grid import RoadGrid
import gym
import numpy as np


class GridDriveV0(gym.Env):
	GRID_DIMENSION				= 30
	MEDIUM_OBS_ROAD_FEATURES 	= 10  # Number of binary ROAD features in Medium Culture
	MEDIUM_OBS_CAR_FEATURES 	= 5  # Number of binary CAR features in Medium Culture (excl. speed)
	MAX_SPEED 					= 100
	MAX_STEP					= 2**7
	
	def __init__(self):
		# Replace here in case culture changes.
		OBS_ROAD_FEATURES	 = self.MEDIUM_OBS_ROAD_FEATURES
		OBS_CAR_FEATURES	 = self.MEDIUM_OBS_CAR_FEATURES

		# Direction (N, S, W, E) + Speed [0-MAX_SPEED]
		self.action_space	   = gym.spaces.MultiDiscrete([4, self.MAX_SPEED])
		self.observation_space = gym.spaces.Tuple([
			gym.spaces.MultiBinary(OBS_ROAD_FEATURES * 4), 	# Extra feature representing whether the cell is accessible.
			gym.spaces.MultiBinary(OBS_CAR_FEATURES),  # Car features
			gym.spaces.MultiDiscrete([self.GRID_DIMENSION, self.GRID_DIMENSION])  # Position
		])
		self.step_counter = 0
		self.keep_grid = False

	def reset(self):
		if not self.keep_grid:
			self.grid = RoadGrid(self.GRID_DIMENSION, self.GRID_DIMENSION, self.MAX_SPEED)
		self.keep_grid = False
		self.step_counter = 0
		return self.get_state()

	def get_state(self):
		return [
			np.array(self.grid.neighbour_features(), dtype=np.int8), 
			np.array(self.grid.agent.binary_features(), dtype=np.int8), 
			np.array(self.grid.agent_position, dtype=np.int64), 
		]

	def step(self, action_vector):
		direction 	= action_vector[0]
		speed 		= action_vector[1]
		reward, explanation = self.grid.move_agent(direction, speed, with_exploratory_bonus=True)
		self.step_counter += 1
		state = self.get_state()
		is_terminal_step = self.step_counter >= self.MAX_STEP #or reward < 0
		return [state, reward, is_terminal_step, {'explanation': explanation}]
