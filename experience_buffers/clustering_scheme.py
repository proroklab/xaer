# -*- coding: utf-8 -*-
import numpy as np
from utils.running_std import RunningMeanStd
from utils.important_information import ImportantInformation
from more_itertools import unique_everseen
from utils.buffer import Buffer, PseudoPrioritizedBuffer
from threading import Lock
import options
flags = options.get()

class none():
	def __init__(self, experience_prioritization_scheme):
		self.experience_prioritization_scheme = experience_prioritization_scheme
		if self.experience_prioritization_scheme:
			self.experience_buffer = PseudoPrioritizedBuffer(
				size=flags.replay_buffer_size, 
				alpha=flags.prioritized_replay_alpha, 
				prioritized_drop_probability=flags.prioritized_drop_probability,
				global_distribution_matching=flags.global_distribution_matching,
				prioritised_cluster_sampling=flags.prioritised_cluster_sampling,
			)
		else:
			self.experience_buffer = Buffer(size=flags.replay_buffer_size)
		ImportantInformation(self.experience_buffer, 'experience_buffer')
		self.experience_buffer_lock = Lock() # Use a locking mechanism to access the buffer because buffers are shared among threads

	def add(self, batch, priority, type_id):
		if self.experience_prioritization_scheme:
			with self.experience_buffer_lock:
				self.experience_buffer.put(batch, priority, type_id)
		else:
			with self.experience_buffer_lock:
				self.experience_buffer.put(batch, type_id)

	def has_atleast(self, n):
		return self.experience_buffer.has_atleast(n)

	def get(self):
		if self.experience_prioritization_scheme:
			with self.experience_buffer_lock:
				return self.experience_buffer.keyed_sample()
		else:
			with self.experience_buffer_lock:
				return (self.experience_buffer.sample(), None, None)

	def update(self, idx, priority, type_id):
		if self.experience_prioritization_scheme:
			with self.experience_buffer_lock:
				self.experience_buffer.update_priority(idx=idx, priority=priority, type_id=type_id)

	@staticmethod
	def is_best_episode(episode_type):
		return episode_type == 'better'

	def get_episode_type(self, episode, agents):
		episode_extrinsic_reward, _ = episode[-1].get_cumulative_reward(agents)
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
	def __init__(self, experience_prioritization_scheme):
		super().__init__(experience_prioritization_scheme)
		self.scaler = RunningMeanStd(batch_size=33)
		ImportantInformation(self.scaler, 'moving_best_extrinsic_reward_scaler')

	def get_episode_type(self, episode, agents):
		episode_extrinsic_reward, _ = episode[-1].get_cumulative_reward(agents)
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
