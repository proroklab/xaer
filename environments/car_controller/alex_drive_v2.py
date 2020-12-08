# -*- coding: utf-8 -*-
from environments.car_controller.car_stuff.alex_discrete.road_grid import RoadGrid
from environments.car_controller.alex_drive_v1 import AlexDriveV1
import gym

class AlexDriveV2(AlexDriveV1):
	def __init__(self):
		# Initialising grid
		self.grid = RoadGrid(self.GRID_DIMENSION, self.GRID_DIMENSION)

		# Replace here in case culture changes.
		OBS_ROAD_FEATURES	 = self.MEDIUM_OBS_ROAD_FEATURES
		OBS_CAR_FEATURES	 = self.MEDIUM_OBS_CAR_FEATURES

		# Direction (N, S, W, E) + Speed [0-200]
		self.action_space	   = gym.spaces.Discrete(4*200)
		self.observation_space = gym.spaces.Tuple([
			gym.spaces.MultiBinary(OBS_ROAD_FEATURES * 4), 	# Extra feature representing whether the cell is accessible.
			gym.spaces.MultiBinary(OBS_CAR_FEATURES),  # Car features
			gym.spaces.MultiDiscrete([self.GRID_DIMENSION, self.GRID_DIMENSION])  # Position
		])
		self.step_counter = 0

	def step(self, action_vector):
		direction 	= action_vector//200
		speed 		= action_vector%200
		reward, explanation = self.grid.move_agent(direction, speed)
		self.step_counter += 1
		state = self.get_state()
		return [state, reward, self.step_counter > 100, {'explanation': explanation}]

