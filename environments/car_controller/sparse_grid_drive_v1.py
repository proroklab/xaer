# -*- coding: utf-8 -*-
from environments.car_controller.grid_drive_v1 import GridDriveV1
import gym

class SparseGridDriveV1(GridDriveV1):

	def reset(self):
		self.cumulated_return = 0
		return super().reset()

	def step(self, action_vector):
		observation, reward, is_terminal_step, info_dict = super().step(action_vector)
		self.cumulated_return += reward
		return [observation, self.cumulated_return if is_terminal_step else 0, is_terminal_step, info_dict]
		