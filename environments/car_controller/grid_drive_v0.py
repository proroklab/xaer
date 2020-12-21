# -*- coding: utf-8 -*-
from environments.car_controller.car_stuff.alex_discrete.road_grid import RoadGrid
import gym
import numpy as np
from environments.car_controller.car_stuff.alex_discrete.road_cultures import EasyRoadCulture, MediumRoadCulture, HardRoadCulture


class GridDriveV0(gym.Env):
	GRID_DIMENSION				= 30
	MAX_SPEED 					= 120
	MAX_STEP					= 2**7
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
			gym.spaces.MultiDiscrete([self.GRID_DIMENSION, self.GRID_DIMENSION]),  # Current Position
			gym.spaces.MultiBinary([self.GRID_DIMENSION, self.GRID_DIMENSION]),  # Visited Cells
			gym.spaces.MultiBinary([self.GRID_DIMENSION, self.GRID_DIMENSION, OBS_ROAD_FEATURES]), 	# Features representing the grid
		])
		self.step_counter = 0
		self.culture = HardRoadCulture({
			'roadworks_ratio':1/8,
			'congestion_charge_ratio':1/8
		})

	def reset(self):
		self.visited_cells = np.zeros((self.GRID_DIMENSION, self.GRID_DIMENSION), dtype=np.int8)
		if self.step_counter%self.MAX_STEP == 0:
			self.grid = RoadGrid(self.GRID_DIMENSION, self.GRID_DIMENSION, self.culture)
			self.grid_features = np.array(self.grid.get_features(), dtype=np.int8)
			self.step_counter = 0
		self.grid.set_random_position()
		x,y = self.grid.agent_position
		self.visited_cells[x][y] = 1
		return self.get_state()

	def get_state(self):
		return (
			np.array(self.grid.neighbour_features(), dtype=np.int8), 
			np.array(self.grid.agent.binary_features(), dtype=np.int8),
			np.array(self.grid.agent_position, dtype=np.int64),
			self.visited_cells,
			self.grid_features, 
		)

	def step(self, action_vector):
		self.step_counter += 1
		direction, speed = action_vector
		can_move, explanation = self.grid.move_agent(direction, speed)
		is_terminal_step = self.step_counter >= self.MAX_STEP #or reward < 0
		x, y = self.grid.agent_position
		reward = 0
		if not can_move: reward = -1
		else:
			if self.visited_cells[x][y] > 0:
				explanation.append('Old cell')
			else: # got ticket in new cell
				reward = (speed+1)/self.MAX_SPEED # in (0,1]
				explanation.append('OK')
		self.visited_cells[x][y] = 1 # do it aftwer checking positions
		return [self.get_state(), reward, is_terminal_step, {'explanation': explanation}]
