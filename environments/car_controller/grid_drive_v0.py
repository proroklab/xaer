# -*- coding: utf-8 -*-
from environments.car_controller.car_stuff.alex_discrete.road_grid import RoadGrid
import gym
import numpy as np


class GridDriveV0(gym.Env):
	GRID_DIMENSION				= 30
	MAX_SPEED 					= 100
	MAX_STEP					= 2**7
	DIRECTIONS					= 5 # N,S,W,E,none
	
	def __init__(self):
		# Replace here in case culture changes.
		OBS_ROAD_FEATURES	 = 10  # Number of binary ROAD features in Medium Culture
		OBS_CAR_FEATURES	 = 5  # Number of binary CAR features in Medium Culture (excl. speed)

		# Direction (N, S, W, E) + Speed [0-MAX_SPEED]
		self.action_space	   = gym.spaces.MultiDiscrete([self.DIRECTIONS, self.MAX_SPEED])
		self.observation_space = gym.spaces.Tuple([ # Current Observation
			gym.spaces.MultiBinary(OBS_ROAD_FEATURES * (self.DIRECTIONS-1)), 	# Extra feature representing whether the cell is accessible.
			gym.spaces.MultiBinary(OBS_CAR_FEATURES),  # Car features
			gym.spaces.MultiDiscrete([self.GRID_DIMENSION, self.GRID_DIMENSION]),  # Current Position
			gym.spaces.MultiBinary([self.GRID_DIMENSION, self.GRID_DIMENSION]),  # Visited Cells
		])
		self.step_counter = 0

	def reset(self):
		self.visited_cells = np.zeros((self.GRID_DIMENSION, self.GRID_DIMENSION), dtype=np.int8)
		if self.step_counter%self.MAX_STEP == 0:
			self.grid = RoadGrid(self.GRID_DIMENSION, self.GRID_DIMENSION)
			self.step_counter = 0
		self.grid.set_random_position()
		x,y = self.grid.agent_position
		self.visited_cells[x][y] = 1
		return self.get_state()

	def get_state(self):
		return [
			np.array(self.grid.neighbour_features(), dtype=np.int8), 
			np.array(self.grid.agent.binary_features(), dtype=np.int8), 
			np.array(self.grid.agent_position, dtype=np.int64), 
			self.visited_cells,
		]

	def step(self, action_vector):
		direction, speed = action_vector
		if direction==self.DIRECTIONS-1: # Terminal action, useful when all neighbours cannot be crossed
			reward = 0
			explanation = 'Terminal action'
			is_terminal_step = True
		else:
			motion_explanation = self.grid.move_agent(direction, speed)
			self.step_counter += 1
			is_terminal_step = self.step_counter >= self.MAX_STEP #or reward < 0
			x, y = self.grid.agent_position
			if motion_explanation:
				reward = -1
				explanation = motion_explanation
			elif self.visited_cells[x][y]>0:
				reward = 0
				explanation = 'Old cell'
			else: # Got ticket in new cell
				reward = (speed+1)/self.MAX_SPEED # in (0,1]
				explanation = 'OK'
			self.visited_cells[x][y] = 1 # do it aftwer checking positions
		return [self.get_state(), reward, is_terminal_step, {'explanation': explanation}]
