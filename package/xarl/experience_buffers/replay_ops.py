from typing import List
import random
import numpy as np
from more_itertools import unique_everseen
from itertools import islice

from ray.util.iter import LocalIterator, _NextValueNotReady
from ray.util.iter_metrics import SharedMetrics
from ray.rllib.execution.replay_buffer import LocalReplayBuffer
from ray.rllib.utils.typing import SampleBatchType


class StoreToReplayBuffer:
	def __init__(self, local_buffer: LocalReplayBuffer = None):
		self.local_actor = local_buffer
		
	def __call__(self, batch: SampleBatchType):
		return self.local_actor.add_batch(batch)

def Replay(local_buffer):
	def gen_replay(_):
		while True:
			item = local_buffer.replay()
			if item is None:
				yield _NextValueNotReady()
			else:
				yield item
	return LocalIterator(gen_replay, SharedMetrics())

class MixInReplay:
	"""This operator adds replay to a stream of experiences.

	It takes input batches, and returns a list of batches that include replayed
	data as well. The number of replayed batches is determined by the
	configured replay proportion. The max age of a batch is determined by the
	number of replay slots.
	"""

	def __init__(self, local_buffer: LocalReplayBuffer, replay_proportion: float):
		"""Initialize MixInReplay.

		Args:
			replay_buffer (Buffer): The replay buffer.
			replay_proportion (float): The input batch will be returned
				and an additional number of batches proportional to this value
				will be added as well.

		Examples:
			# replay proportion 2:1
			>>> replay_op = MixInReplay(rollouts, 100, replay_proportion=2)
			>>> print(next(replay_op))
			[SampleBatch(<input>), SampleBatch(<replay>), SampleBatch(<rep.>)]

			# replay proportion 0:1, replay disabled
			>>> replay_op = MixInReplay(rollouts, 100, replay_proportion=0)
			>>> print(next(replay_op))
			[SampleBatch(<input>)]
		"""
		self.replay_buffer = local_buffer
		self.replay_proportion = replay_proportion

	def __call__(self, sample_batch):
		output_batches = []
		if self.replay_buffer.can_replay():
			f = self.replay_proportion
			while random.random() < f:
				f -= 1
				replayed_batch = self.replay_buffer.replay()
				if replayed_batch:
					output_batches.append(replayed_batch)
		# Put in the experience buffer, after replaying, to avoid double sampling.
		sample_batch = self.replay_buffer.add_batch(sample_batch)
		output_batches.append(sample_batch)
		return output_batches
