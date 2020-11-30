# -*- coding: utf-8 -*-
import gym

class AlexDriveV1(gym.Env):
	
	def __init__(self):
		# steering angle, and speed
		self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
		# observations
		self.observation_space = gym.spaces.Tuple([
			gym.spaces.Box(**shape, dtype=np.float32) 
			for shape in self.get_state_shape()
		])

	def get_state_shape(self):
		return [
			{
				'low': -15,
				'high': 15,
				'shape': (1, 2, 2), # current road view: relative coordinates of road.start.pos and road.end.pos
			},
			{
				'low': -15,
				'high': 15,
				'shape': ( # closest junctions view
					2, # number of junctions close to current road
					Junction.max_roads_connected, 
					1+1, # relative heading vector + road colour
				),
			}
		]

	def reset(self):
		pass

	def step(self, action_vector):
		pass
