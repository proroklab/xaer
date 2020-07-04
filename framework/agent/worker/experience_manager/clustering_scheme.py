# -*- coding: utf-8 -*-
import numpy as np
from utils.running_std import RunningMeanStd
from utils.important_information import ImportantInformation
from more_itertools import unique_everseen

class none():
	@staticmethod
	def is_best(episode_type):
		return episode_type == 'better'

	def get_episode_type(self, episode, agents):
		episode_extrinsic_reward, _ = episode.get_cumulative_reward(agents)
		return 'better' if episode_extrinsic_reward > 0 else 'worse' # Best batches = batches that lead to positive extrinsic reward

	def get_batch_type(self, batch, agents, episode_type):
		return 'none'

class extrinsic_reward(none):
	def get_batch_type(self, batch, agents, episode_type):
		# Build batch type
		batch_extrinsic_reward, _ = batch.get_cumulative_reward(agents)
		#=======================================================================
		# if batch_extrinsic_reward > 0:
		# 	print("Adding new batch with reward: extrinsic {}, intrinsic {}".format(batch_extrinsic_reward, batch_intrinsic_reward))
		#=======================================================================
		batch_type = 'positive' if batch_extrinsic_reward > 0 else 'negative'
		return f"{episode_type}/{batch_type}"

class moving_best_extrinsic_reward(extrinsic_reward):
	def __init__(self):
		self.scaler = RunningMeanStd(batch_size=33)
		ImportantInformation(self.scaler, 'moving_best_extrinsic_reward_scaler')

	def get_episode_type(self, episode, agents):
		episode_extrinsic_reward, _ = episode.get_cumulative_reward(agents)
		self.scaler.update([episode_extrinsic_reward])
		return 'better' if episode_extrinsic_reward > self.scaler.mean else 'worse'
		
class moving_best_extrinsic_reward_with_type(moving_best_extrinsic_reward):
	def get_batch_type(self, batch, agents, episode_type):
		batch_type = '-'.join(sorted(unique_everseen(batch.get_all_actions(actions=['reward_types'], agents=agents)[0])))
		# print(batch_type)
		return f"{episode_type}/{batch_type}"

class reward_with_type(none):
	def get_batch_type(self, batch, agents, episode_type):
		batch_type = '-'.join(sorted(unique_everseen(batch.get_all_actions(actions=['reward_types'], agents=agents)[0])))
		return batch_type
