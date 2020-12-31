# -*- coding: utf-8 -*-
from environments.car_controller.grid_drive_v0 import GridDriveV0
import gym

class GridDriveV1(GridDriveV0):
	def __init__(self):
		super().__init__()
		# Direction (N, S, W, E) + Speed [0-200]
		self.action_space	   = gym.spaces.Discrete(self.DIRECTIONS*self.MAX_GAPPED_SPEED)

	def step(self, action_vector):
		direction = action_vector//self.MAX_GAPPED_SPEED
		gapped_speed = action_vector%self.MAX_GAPPED_SPEED
		return super().step((direction,gapped_speed))
