# -*- coding: utf-8 -*-
import numpy as np

class GameWrapper(object):
	state_scaler = 1

	def get_concatenation_size(self):
		return sum(map(lambda x: x[0], self.get_action_shape()))+1
		
	# Last Action-Reward: Jaderberg, Max, et al. "Reinforcement learning with unsupervised auxiliary tasks." arXiv preprint arXiv:1611.05397 (2016).
	def get_concatenation(self):
		if self.last_action is None:
			return np.zeros(self.get_concatenation_size())
		flatten_action = np.concatenate([np.reshape(a,-1) for a in self.last_action], -1)
		return np.concatenate((flatten_action,[self.last_reward]), -1)

	def process(self, action):
		pass

	def reset(self):
		self.is_over = False
		self.episode_statistics = {}
		self.step = 0
	
	def get_state_shape(self):
		pass

	def get_action_shape(self):
		pass
	
	def get_test_result(self):
		return None
		
	def get_screen_shape(self):
		return self.get_state_shape()
	
	def get_info(self):
		return None
	
	def get_screen(self):
		return None

	def has_masked_actions(self):
		return False
	
	def get_statistics(self):
		return {}
	
