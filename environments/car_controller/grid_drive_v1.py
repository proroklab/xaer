# -*- coding: utf-8 -*-
from environments.car_controller.grid_drive_v0 import GridDriveV0
import gym

class GridDriveV1(GridDriveV0):
	def __init__(self):
		super().__init__()
		# Direction (N, S, W, E) + Speed [0-200]
		self.action_space	   = gym.spaces.Discrete(4*200)

	def step(self, action_vector):
		direction 	= action_vector//200
		speed 		= action_vector%200
		reward, explanation = self.grid.move_agent(direction, speed, with_exploratory_bonus=False)
		self.step_counter += 1
		state = self.get_state()
		return [state, reward, self.step_counter >= 2**7, {'explanation': explanation}]
