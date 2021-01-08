# -*- coding: utf-8 -*-
import numpy as np
from collections import Counter
from more_itertools import unique_everseen
from xarl.utils.running_statistics import RunningStats
import itertools

class none():
	def get_episode_type(self, episode):
		return 'none'

	def get_batch_type(self, batch, episode_type='none'):
		return [(episode_type,'none')]

class reward_against_zero(none):
	def get_episode_type(self, episode):
		episode_extrinsic_reward = sum((np.sum(batch["rewards"]) for batch in episode))
		# episode_extrinsic_reward = np.sum(episode[-1]["rewards"])
		return 'better' if episode_extrinsic_reward > 0 else 'worse' # Best batches = batches that lead to positive extrinsic reward

	def get_batch_type(self, batch, episode_type='none'):
		batch_extrinsic_reward = np.sum(batch["rewards"])
		batch_type = 'greater' if batch_extrinsic_reward > 0 else 'lower'
		return [(episode_type, batch_type)]

class reward_against_mean(none):
	def __init__(self):
		self.episode_stats = RunningStats(window_size=2**6)
		self.batch_stats = RunningStats(window_size=2**10)

	def get_episode_type(self, episode):
		episode_extrinsic_reward = sum((np.sum(batch["rewards"]) for batch in episode))
		# episode_extrinsic_reward = np.sum(episode[-1]["rewards"])
		self.episode_stats.push(episode_extrinsic_reward)
		return 'better' if episode_extrinsic_reward > self.episode_stats.mean else 'worse'

	def get_batch_type(self, batch, episode_type='none'):
		batch_extrinsic_reward = np.sum(batch["rewards"])
		self.batch_stats.push(batch_extrinsic_reward)
		batch_type = 'greater' if batch_extrinsic_reward > self.batch_stats.mean else 'lower'
		return [(episode_type, batch_type)]
		
class multiple_types_with_reward_against_mean(reward_against_mean):
	def get_batch_type(self, batch, episode_type='none'):
		batch_type = super().get_batch_type(batch, episode_type)[0][-1]
		explanation_iter = map(lambda x: x.get("explanation",'None'), batch["infos"])
		explanation_iter = map(lambda x: x if isinstance(x,(list,tuple)) else [x], explanation_iter)
		explanation_iter = itertools.chain(*explanation_iter)
		explanation_iter = unique_everseen(explanation_iter)
		explanation_iter = map(lambda x:(episode_type, batch_type, x), explanation_iter)
		return tuple(explanation_iter)

class multiple_types_with_reward_against_zero(reward_against_zero):
	def get_batch_type(self, batch, episode_type='none'):
		batch_type = super().get_batch_type(batch, episode_type)[0][-1]
		explanation_iter = map(lambda x: x.get("explanation",'None'), batch["infos"])
		explanation_iter = map(lambda x: x if isinstance(x,(list,tuple)) else [x], explanation_iter)
		explanation_iter = itertools.chain(*explanation_iter)
		explanation_iter = unique_everseen(explanation_iter)
		explanation_iter = map(lambda x:(episode_type, batch_type, x), explanation_iter)
		return tuple(explanation_iter)

class type_with_reward_against_mean(multiple_types_with_reward_against_mean):
	def get_batch_type(self, batch, episode_type='none'):
		explanation_iter = super().get_batch_type(batch, episode_type)
		batch_type = explanation_iter[0][-2]
		explanation_iter = map(lambda x:x[-1], explanation_iter)
		return [(episode_type, batch_type, sorted(explanation_iter))]

class multiple_types(reward_against_mean):
	def get_batch_type(self, batch, episode_type='none'):
		explanation_iter = map(lambda x: x.get("explanation",'None'), batch["infos"])
		explanation_iter = map(lambda x: x if isinstance(x,(list,tuple)) else [x], explanation_iter)
		explanation_iter = itertools.chain(*explanation_iter)
		explanation_iter = unique_everseen(explanation_iter)
		explanation_iter = map(lambda x:(episode_type, x), explanation_iter)
		return tuple(explanation_iter)

class type(multiple_types):
	def get_batch_type(self, batch, episode_type='none'):
		explanation_iter = super().get_batch_type(batch, episode_type)
		return [(episode_type, sorted(explanation_iter))]
