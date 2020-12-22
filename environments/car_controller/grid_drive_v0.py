# -*- coding: utf-8 -*-
from environments.car_controller.car_stuff.alex_discrete.road_grid import RoadGrid
import gym
import numpy as np
from environments.car_controller.car_stuff.alex_discrete.road_cultures import EasyRoadCulture, MediumRoadCulture, HardRoadCulture


class GridDriveV0(gym.Env):
	GRID_DIMENSION				= 15
	MAX_SPEED 					= 120
	MAX_STEP					= 2**5
	DIRECTIONS					= 4 # N,S,W,E
	
	def __init__(self):
		# Replace here in case culture changes.
		OBS_ROAD_FEATURES	 = 10  # Number of binary ROAD features in Hard Culture
		OBS_CAR_FEATURES	 = 5  # Number of binary CAR features in Hard Culture (excl. speed)

		# Direction (N, S, W, E) + Speed [0-MAX_SPEED]
		self.action_space	   = gym.spaces.MultiDiscrete([self.DIRECTIONS, self.MAX_SPEED])
		self.observation_space = gym.spaces.Tuple([ # Current Observation
			gym.spaces.MultiBinary(OBS_ROAD_FEATURES * self.DIRECTIONS), # Neighbourhood view
			gym.spaces.MultiBinary(OBS_CAR_FEATURES),  # Car features
			gym.spaces.MultiBinary([self.GRID_DIMENSION, self.GRID_DIMENSION, OBS_ROAD_FEATURES+2]), # Features representing the grid + visited cells + current position
		])
		self.step_counter = 0
		self.culture = HardRoadCulture({
			'roadworks_ratio': 1/8,
			'congestion_charge_ratio': 1/8,
		})

	def reset(self):
		if self.step_counter%self.MAX_STEP == 0:
			self.grid = RoadGrid(self.GRID_DIMENSION, self.GRID_DIMENSION, self.culture)
			self.grid_features = np.array(self.grid.get_features(), dtype=np.int8)
			self.step_counter = 0
		self.grid_view = np.concatenate([
			self.grid_features,
			np.zeros((self.GRID_DIMENSION, self.GRID_DIMENSION, 2), dtype=np.int8), # current position + visited cells
		], -1)
		self.grid.set_random_position()
		x,y = self.grid.agent_position
		self.grid_view[x][y][-2] = 1 # set current cell as visited
		return self.get_state()

	def get_state(self):
		return (
			np.array(self.grid.neighbour_features(), dtype=np.int8), 
			np.array(self.grid.agent.binary_features(), dtype=np.int8),
			self.grid_view,
		)

	def step(self, action_vector):
		self.step_counter += 1
		x, y = self.grid.agent_position
		self.grid_view[x][y][-1] = 0 # remove old position
		direction, speed = action_vector
		can_move, explanation = self.grid.move_agent(direction, speed)
		is_terminal_step = self.step_counter >= self.MAX_STEP #or reward < 0
		x, y = self.grid.agent_position
		if not can_move:
			reward = -(speed+1)/self.MAX_SPEED # in [-1,0)
			is_terminal_step = True
		else:
			if self.grid_view[x][y][-2] > 0: # already visited cell
				reward = 0
				explanation.append('Old cell')
			else:
				reward = (speed+1)/self.MAX_SPEED # in (0,1]
				explanation.append('OK')
		# do it aftwer checking positions
		self.grid_view[x][y][-2] = 1 # set current cell as visited
		self.grid_view[x][y][-1] = 1 # set new position
		return [self.get_state(), reward, is_terminal_step, {'explanation': explanation}]
