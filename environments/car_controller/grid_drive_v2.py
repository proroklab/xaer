# -*- coding: utf-8 -*-
from environments.car_controller.grid_drive_v0 import GridDriveV0
import gym

class GridDriveV2(GridDriveV0):
	def __init__(self):
		super().__init__()
		# Direction (N, S, W, E) + Speed [0-200]
		self.action_space	   = gym.spaces.Discrete(4*self.MAX_SPEED)

	def step(self, action_vector):
		direction 	= action_vector//self.MAX_SPEED
		speed 		= action_vector%self.MAX_SPEED
		reward, explanation = self.grid.move_agent(direction, speed, with_exploratory_bonus=False)
		self.step_counter += 1
		state = self.get_state()
		is_terminal_step = self.step_counter >= self.MAX_STEP or reward < 0
		if reward < 0:
			self.keep_grid = True
		return [state, reward, is_terminal_step, {'explanation': explanation}]
