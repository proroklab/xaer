# -*- coding: utf-8 -*-
from environments.car_controller.car_stuff.alex_discrete.road_grid import RoadGrid
import gym


class AlexDriveV1(gym.Env):
	GRID_DIMENSION				= 30
	MEDIUM_OBS_ROAD_FEATURES 	= 6  # Number of binary ROAD features in Medium Culture
	MEDIUM_OBS_CAR_FEATURES 	= 1  # Number of binary CAR features in Medium Culture (excl. speed)
	
	def __init__(self):
		# Initialising grid
		self.grid = RoadGrid(self.GRID_DIMENSION, self.GRID_DIMENSION)

		# Replace here in case culture changes.
		OBS_ROAD_FEATURES	 = self.MEDIUM_OBS_ROAD_FEATURES
		OBS_CAR_FEATURES	 = self.MEDIUM_OBS_CAR_FEATURES

		# Direction (N, S, W, E) + Speed [0-200]
		self.action_space	   = gym.spaces.MultiDiscrete([4, 200])
		self.observation_space = gym.spaces.Tuple([
			gym.spaces.MultiBinary((1 + OBS_ROAD_FEATURES) * 4), 	# Extra feature representing
																	# whether the cell is accessible.
			gym.spaces.MultiBinary(OBS_CAR_FEATURES),  # Car features
			gym.spaces.MultiDiscrete([self.GRID_DIMENSION, self.GRID_DIMENSION])  # Position
		])
		self.step_counter = 0

	def reset(self):
		self.grid = RoadGrid(self.GRID_DIMENSION, self.GRID_DIMENSION)
		self.step_counter = 0

	def step(self, action_vector):
		direction 	= action_vector[0]
		speed 		= action_vector[1]
		reward, explanation = self.grid.move_agent(direction, speed)
		self.step_counter += 1
		state = self.grid.neighbour_features()
		return [state, reward, self.step_counter > 100, {'explanation': explanation}]




