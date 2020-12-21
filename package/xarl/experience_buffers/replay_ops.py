from typing import List
import random
import numpy as np
from more_itertools import unique_everseen
from itertools import islice

from ray.util.iter import LocalIterator, _NextValueNotReady
from ray.util.iter_metrics import SharedMetrics
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch, DEFAULT_POLICY_ID

from xarl.experience_buffers.replay_buffer import LocalReplayBuffer
from xarl.experience_buffers.clustering_scheme import *

def get_clustered_replay_buffer(config):
	assert config["batch_mode"] == "complete_episodes" or not eval(config["clustering_scheme"]).batch_type_is_based_on_episode_type, f"This algorithm requires 'complete_episodes' as batch_mode when 'clustering_scheme' is {config['clustering_scheme']}"
	local_replay_buffer = LocalReplayBuffer(
		prioritized_replay=config["prioritized_replay"],
		buffer_options=config["buffer_options"], 
		learning_starts=config["learning_starts"], 
		update_only_sampled_cluster=config["update_only_sampled_cluster"],
	)
	clustering_scheme = eval(config["clustering_scheme"])()
	return local_replay_buffer, clustering_scheme

def assign_types(batch, clustering_scheme, replay_sequence_length):
	batch_list = []
	for episode in batch.split_by_episode():
		sub_batch_list = episode.timeslices(replay_sequence_length)
		episode_type = clustering_scheme.get_episode_type(sub_batch_list)
		# print(episode_type)
		for sub_batch in sub_batch_list:
			sub_batch_type = clustering_scheme.get_batch_type(sub_batch, episode_type)
			sub_batch["infos"][0]['batch_type'] = sub_batch_type
			sub_batch["infos"][0]['batch_index'] = {}
		batch_list += sub_batch_list
	return batch_list

class StoreToReplayBuffer:
	def __init__(self, local_buffer: LocalReplayBuffer = None):
		self.local_actor = local_buffer
		
	def __call__(self, batch: SampleBatchType):
		return self.local_actor.add_batch(batch)

def Replay(local_buffer, replay_batch_size=1, filter_duplicates=False):
	def gen_replay(_):
		while True:
			batch_list = local_buffer.replay(replay_size=replay_batch_size, filter_duplicates=filter_duplicates)
			if not batch_list:
				yield _NextValueNotReady()
			else:
				for batch in batch_list:
					yield batch
	return LocalIterator(gen_replay, SharedMetrics())

class MixInReplay:
	"""This operator adds replay to a stream of experiences.

	It takes input batches, and returns a list of batches that include replayed
	data as well. The number of replayed batches is determined by the
	configured replay proportion. The max age of a batch is determined by the
	number of replay slots.
	"""

	def __init__(self, local_buffer, replay_proportion, update_replayed_fn=None, filter_duplicates=False):
		self.replay_buffer = local_buffer
		self.replay_proportion = replay_proportion
		self.update_replayed_fn = update_replayed_fn
		self.filter_duplicates = filter_duplicates

	def __call__(self, sample_batch):
		def get_updated_batch(batch):
			if isinstance(batch, MultiAgentBatch):
				batch.policy_batches = {
					k:self.update_replayed_fn(v)
					for k,v in batch.policy_batches.items()
				}
				return batch
			return self.update_replayed_fn(batch)
		# Put in the experience buffer
		sample_batch = self.replay_buffer.add_batch(sample_batch) # allow for duplicates in output_batches
		output_batches = [sample_batch]
		if self.replay_buffer.can_replay():
			n = np.random.poisson(self.replay_proportion)
			if n > 0:
				batch_list = self.replay_buffer.replay(replay_size=n, filter_duplicates=self.filter_duplicates) # allow for duplicates in output_batches
				if self.update_replayed_fn:
					batch_list = list(map(get_updated_batch, batch_list))
				output_batches += batch_list
		return output_batches
