# -*- coding: utf-8 -*-
from environments.car_controller.car_stuff.alex_discrete.road_grid import RoadGrid
import gym
import numpy as np


class GridDriveV0(gym.Env):
	GRID_DIMENSION				= 30
	MAX_SPEED 					= 100
	MAX_STEP					= 2**7
	DIRECTIONS					= 4 # N,S,W,E,none
	
	def __init__(self):
		# Replace here in case culture changes.
		OBS_ROAD_FEATURES	 = 10  # Number of binary ROAD features in Medium Culture
		OBS_CAR_FEATURES	 = 5  # Number of binary CAR features in Medium Culture (excl. speed)

		# Direction (N, S, W, E) + Speed [0-MAX_SPEED]
		self.action_space	   = gym.spaces.MultiDiscrete([self.DIRECTIONS, self.MAX_SPEED])
		self.observation_space = gym.spaces.Tuple([ # Current Observation
			gym.spaces.MultiBinary([self.GRID_DIMENSION, self.GRID_DIMENSION, OBS_ROAD_FEATURES]), 	# Features representing the grid
			gym.spaces.MultiBinary(OBS_CAR_FEATURES),  # Car features
			gym.spaces.MultiDiscrete([self.GRID_DIMENSION, self.GRID_DIMENSION]),  # Current Position
			gym.spaces.MultiBinary([self.GRID_DIMENSION, self.GRID_DIMENSION]),  # Visited Cells
			gym.spaces.MultiBinary(1),  # found_initial_state
		])
		self.step_counter = 0

	def reset(self):
		self.visited_cells = np.zeros((self.GRID_DIMENSION, self.GRID_DIMENSION), dtype=np.int8)
		if self.step_counter%self.MAX_STEP == 0:
			self.grid = RoadGrid(self.GRID_DIMENSION, self.GRID_DIMENSION)
			self.grid_features = np.array(self.grid.get_features(), dtype=np.int8)
			self.step_counter = 0
		self.grid.set_random_position()
		x,y = self.grid.agent_position
		self.visited_cells[x][y] = 1
		self.found_initial_state = False
		return self.get_state()

	def get_state(self):
		return (
			self.grid_features, 
			np.array(self.grid.agent.binary_features(), dtype=np.int8),
			np.array(self.grid.agent_position, dtype=np.int64),
			self.visited_cells,
			np.array([self.found_initial_state], dtype=np.int8),
		)

	def step(self, action_vector):
		self.step_counter += 1
		direction, speed = action_vector
		can_move, explanation = self.grid.move_agent(direction, speed)
		is_terminal_step = self.step_counter >= self.MAX_STEP #or reward < 0
		x, y = self.grid.agent_position
		reward = 0
		if not can_move:
			if self.found_initial_state: # first, let the agent find an initial state where it can move
				reward = -1
		else:
			self.found_initial_state = True # an initial state where the agent can move is found 
			if self.visited_cells[x][y] > 0:
				explanation.append('Old cell')
			else: # got ticket in new cell
				reward = (speed+1)/self.MAX_SPEED # in (0,1]
				explanation.append('OK')
		self.visited_cells[x][y] = 1 # do it aftwer checking positions
		return [self.get_state(), reward, is_terminal_step, {'explanation': explanation}]
