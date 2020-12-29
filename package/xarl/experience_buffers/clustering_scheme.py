# -*- coding: utf-8 -*-
import numpy as np
from collections import Counter
from more_itertools import unique_everseen
from xarl.utils.running_std import RunningMeanStd
import itertools

class none():
	batch_type_is_based_on_episode_type = False

	def get_episode_type(self, episode):
		return 'none'

	def get_batch_type(self, batch, episode_type):
		return 'none'

class extrinsic_reward(none):
	batch_type_is_based_on_episode_type = True

	@staticmethod
	def is_best_episode(episode_type):
		return episode_type == 'better'

	def get_episode_type(self, episode):
		episode_extrinsic_reward = sum((np.sum(batch["rewards"]) for batch in episode))
		return 'better' if episode_extrinsic_reward > 0 else 'worse' # Best batches = batches that lead to positive extrinsic reward

	def get_batch_type(self, batch, episode_type):
		# Build batch type
		batch_extrinsic_reward = np.sum(batch["rewards"])
		#=======================================================================
		# if batch_extrinsic_reward > 0:
		# 	print("Adding new batch with reward: extrinsic {}, intrinsic {}".format(batch_extrinsic_reward, batch_intrinsic_reward))
		#=======================================================================
		batch_type = 'positive' if batch_extrinsic_reward > 0 else 'negative'
		return '{}-{}'.format(episode_type,batch_type)

class moving_best_extrinsic_reward(extrinsic_reward):
	def __init__(self):
		self.scaler = RunningMeanStd(batch_size=33)

	def get_episode_type(self, episode):
		episode_extrinsic_reward = sum((np.sum(batch["rewards"]) for batch in episode))
		self.scaler.update([episode_extrinsic_reward])
		return 'better' if episode_extrinsic_reward > self.scaler.mean else 'worse'
		
class moving_best_extrinsic_reward_with_type(moving_best_extrinsic_reward):
	def get_batch_type(self, batch, episode_type):
		explanation_iter = map(lambda x: x.get("explanation",'None'), batch["infos"])
		explanation_iter = map(lambda x: x if isinstance(x,(list,tuple)) else [x], explanation_iter)
		explanation_iter = itertools.chain(*explanation_iter)
		explanation_iter = unique_everseen(explanation_iter)
		return '{}-{}'.format(episode_type,'-'.join(sorted(explanation_iter)))

class moving_best_extrinsic_reward_with_multiple_types(moving_best_extrinsic_reward):
	def get_batch_type(self, batch, episode_type):
		explanation_iter = map(lambda x: x.get("explanation",'None'), batch["infos"])
		explanation_iter = map(lambda x: x if isinstance(x,(list,tuple)) else [x], explanation_iter)
		explanation_iter = itertools.chain(*explanation_iter)
		explanation_iter = unique_everseen(explanation_iter)
		explanation_iter = map(lambda x:'{}-{}'.format(episode_type,x), explanation_iter)
		return tuple(explanation_iter)

class reward_with_type(none):
	def get_batch_type(self, batch, episode_type):
		explanation_iter = map(lambda x: x.get("explanation",'None'), batch["infos"])
		explanation_iter = map(lambda x: x if isinstance(x,(list,tuple)) else [x], explanation_iter)
		explanation_iter = itertools.chain(*explanation_iter)
		explanation_iter = unique_everseen(explanation_iter)
		return '-'.join(sorted(explanation_iter))

class reward_with_multiple_types(none):
	def get_batch_type(self, batch, episode_type):
		explanation_iter = map(lambda x: x.get("explanation",'None'), batch["infos"])
		explanation_iter = map(lambda x: x if isinstance(x,(list,tuple)) else [x], explanation_iter)
		explanation_iter = itertools.chain(*explanation_iter)
		explanation_iter = unique_everseen(explanation_iter)
		return tuple(explanation_iter)
