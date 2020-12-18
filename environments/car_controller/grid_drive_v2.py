# -*- coding: utf-8 -*-
from environments.car_controller.grid_drive_v1 import GridDriveV1
import gym

class GridDriveV2(GridDriveV1):

	def step(self, action_vector):
		state, reward, is_terminal_step, info_dict = super().step(action_vector)
		if reward < 0:
			is_terminal_step = True
			self.keep_grid = True
		return [state, reward, is_terminal_step, info_dict]
