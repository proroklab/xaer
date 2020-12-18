# -*- coding: utf-8 -*-
from environments.car_controller.grid_drive_v0 import GridDriveV0
import gym

class GridDriveV1(GridDriveV0):
	def __init__(self):
		super().__init__()
		# Direction (N, S, W, E) + Speed [0-200]
		self.action_space	   = gym.spaces.Discrete(4*self.MAX_SPEED)

	def step(self, action_vector):
		direction 	= action_vector//self.MAX_SPEED
		speed 		= action_vector%self.MAX_SPEED
		return super().step((direction,speed))
