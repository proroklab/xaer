import gym
from gym.utils import seeding
import numpy as np

import logging
logger = logging.getLogger(__name__)

class Example_v0(gym.Env):
	LF_MIN = 1
	RT_MAX = 10
	MOVE_LF = 0
	MOVE_RT = 1
	MAX_STEPS = 10
	REWARD_AWAY = -2
	REWARD_STEP = -1
	REWARD_GOAL = MAX_STEPS
	metadata = {
		"render.modes": ["human"]
	}

	def seed(self, seed=None):
		logger.warning(f"Setting random seed to: {seed}")
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def __init__ (self, config):
		self.action_space = gym.spaces.Discrete(2)
		self.observation_space = gym.spaces.Discrete(self.RT_MAX + 1)
		# possible positions to chose on `reset()`
		self.goal = int((self.LF_MIN + self.RT_MAX - 1) / 2)
		self.init_positions = list(range(self.LF_MIN, self.RT_MAX))
		self.init_positions.remove(self.goal)

	def reset (self):
		self.position = self.np_random.choice(self.init_positions)
		self.count = 0
		self.state = self.position
		self.reward = 0
		self.done = False
		self.info = None
		return self.state

	def step (self, action):
		self.info = {}
		if self.done:
			# should never reach this point
			print("EPISODE DONE!!!")
		elif self.count == self.MAX_STEPS:
			self.done = True;
		else:
			assert self.action_space.contains(action)
			self.count += 1
			# logic to handle an action ...
			if action == self.MOVE_LF:
				if self.position == self.LF_MIN:
					# invalid
					self.reward = self.REWARD_AWAY
					self.info['explanation'] = 'REWARD_AWAY'
				else:
					self.position -= 1
				if self.position == self.goal:
					# on goal now
					self.reward = self.REWARD_GOAL
					self.info['explanation'] = 'REWARD_GOAL'
					self.done = 1
				elif self.position < self.goal:
					# moving away from goal
					self.reward = self.REWARD_AWAY
					self.info['explanation'] = 'REWARD_AWAY'
				else:
					# moving toward goal
					self.reward = self.REWARD_STEP
					self.info['explanation'] = 'REWARD_STEP'
			elif action == self.MOVE_RT:
				if self.position == self.RT_MAX:
					# invalid
					self.reward = self.REWARD_AWAY
					self.info['explanation'] = 'REWARD_AWAY'
				else:
					self.position += 1
				if self.position == self.goal:
					# on goal now
					self.reward = self.REWARD_GOAL
					self.info['explanation'] = 'REWARD_GOAL'
					self.done = 1
				elif self.position > self.goal:
					# moving away from goal
					self.reward = self.REWARD_AWAY
					self.info['explanation'] = 'REWARD_AWAY'
				else:
					# moving toward goal
					self.reward = self.REWARD_STEP
					self.info['explanation'] = 'REWARD_STEP'
			self.state = self.position
			self.info["dist"] = self.goal - self.position
		try:
			assert self.observation_space.contains(self.state)
		except AssertionError:
			print("INVALID STATE", self.state)
		return [self.state, self.reward, self.done, self.info]

	def render (self, mode="human"):
		s = "position: {:2d}  reward: {:2d}  info: {}"
		print(s.format(self.state, self.reward, self.info))

	def close (self):
		pass
