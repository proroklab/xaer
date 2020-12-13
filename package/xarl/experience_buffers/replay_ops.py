from typing import List
import random
import numpy as np
from more_itertools import unique_everseen
from itertools import islice

from ray.util.iter import LocalIterator, _NextValueNotReady
from ray.util.iter_metrics import SharedMetrics
from ray.rllib.execution.replay_buffer import LocalReplayBuffer
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch, DEFAULT_POLICY_ID


class StoreToReplayBuffer:
	def __init__(self, local_buffer: LocalReplayBuffer = None):
		self.local_actor = local_buffer
		
	def __call__(self, batch: SampleBatchType):
		return self.local_actor.add_batch(batch)

def Replay(local_buffer):
	def gen_replay(_):
		while True:
			batch_list = local_buffer.replay()
			if batch_list is None:
				yield _NextValueNotReady()
			else:
				yield MultiAgentBatch.concat_samples(batch_list)
	return LocalIterator(gen_replay, SharedMetrics())

class MixInReplay:
	"""This operator adds replay to a stream of experiences.

	It takes input batches, and returns a list of batches that include replayed
	data as well. The number of replayed batches is determined by the
	configured replay proportion. The max age of a batch is determined by the
	number of replay slots.
	"""

	def __init__(self, local_buffer, replay_proportion, update_replayed_fn=None):
		self.replay_buffer = local_buffer
		self.replay_proportion = replay_proportion
		self.update_replayed_fn = update_replayed_fn

	def __call__(self, sample_batch):
		def get_updated_batch(batch):
			if isinstance(batch, MultiAgentBatch):
				batch.policy_batches = {
					k:self.update_replayed_fn(v)
					for k,v in batch.policy_batches.items()
				}
				return batch
			return self.update_replayed_fn(batch)
		output_batches = []
		if self.replay_buffer.can_replay():
			f = self.replay_proportion
			times_to_replay = f//1
			if times_to_replay!=f and random.random() < f - times_to_replay:
				times_to_replay += 1
			if times_to_replay > 0:
				batch_list = self.replay_buffer.replay(replay_size=int(times_to_replay), filter_unique=True)
				if self.update_replayed_fn:
					batch_list = list(map(get_updated_batch, batch_list))
				output_batches += batch_list
		# Put in the experience buffer, after replaying, to avoid double sampling.
		sample_batch = self.replay_buffer.add_batch(sample_batch)
		output_batches.append(sample_batch)
		return output_batches
