import collections
import logging
import numpy as np
import platform
from more_itertools import unique_everseen
from itertools import islice

# Import ray before psutil will make sure we use psutil's bundled version
import ray  # noqa F401
import psutil  # noqa E402

from xarl.experience_buffers.buffer.pseudo_prioritized_buffer import PseudoPrioritizedBuffer
from xarl.experience_buffers.buffer.buffer import Buffer

from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch, DEFAULT_POLICY_ID
from ray.util.iter import ParallelIteratorWorker
from ray.rllib.utils.timer import TimerStat

logger = logging.getLogger(__name__)

class LocalReplayBuffer(ParallelIteratorWorker):
	"""A replay buffer shard.

	Ray actors are single-threaded, so for scalability multiple replay actors
	may be created to increase parallelism."""

	def __init__(self, 
		prioritized_replay=True,
		buffer_options=None, 
		learning_starts=1000, 
		replay_sequence_length=None,
		replay_batch_size=1,
	):
		self.replay_batch_size = replay_batch_size
		self.replay_sequence_length = replay_sequence_length
		if replay_sequence_length and replay_sequence_length > 1:
			self.replay_batch_size = int(max(1, replay_batch_size // replay_sequence_length))
			logger.info("Since replay_sequence_length={} and replay_batch_size={}, we will replay {} sequences at a time.".format(replay_sequence_length, replay_batch_size, self.replay_batch_size))
		self.prioritized_replay = prioritized_replay
		self.buffer_options = {} if not buffer_options else buffer_options
		self.replay_starts = learning_starts

		def gen_replay():
			while True:
				yield self.replay()

		ParallelIteratorWorker.__init__(self, gen_replay, False)

		def new_buffer():
			return PseudoPrioritizedBuffer(**self.buffer_options) if self.prioritized_replay else Buffer(self.buffer_options['size'])

		self.replay_buffers = collections.defaultdict(new_buffer)

		# Metrics
		self.add_batch_timer = TimerStat()
		self.replay_timer = TimerStat()
		self.update_priorities_timer = TimerStat()
		self.num_added = 0

	def get_host(self):
		return platform.node()

	def add_batch(self, batch):
		# Make a copy so the replay buffer doesn't pin plasma memory.
		batch = batch.copy()
		batch_type = batch['infos'][0]["batch_type"]
		# Handle everything as if multiagent
		if isinstance(batch, SampleBatch):
			batch = MultiAgentBatch({DEFAULT_POLICY_ID: batch}, batch.count)
		with self.add_batch_timer:
			for policy_id, b in batch.policy_batches.items():
				if not self.replay_sequence_length:
					self.num_added += 1
					if isinstance(batch_type,(tuple,list)):
						for sub_batch_type in batch_type:
							self.replay_buffers[policy_id].add(b, sub_batch_type)
					else:
						self.replay_buffers[policy_id].add(b, batch_type)
				else:
					b_infos = b['infos'][0]
					for s in b.timeslices(self.replay_sequence_length):
						self.num_added += 1
						s_infos = s['infos'][0]
						if "batch_type" not in s_infos:
							s_infos["batch_type"] = batch_type
						if "batch_index" not in s_infos:
							s_infos["batch_index"] = {}
						if isinstance(batch_type,(tuple,list)):
							for sub_batch_type in batch_type:
								self.replay_buffers[policy_id].add(s, sub_batch_type)
						else:
							self.replay_buffers[policy_id].add(s, batch_type)
		return batch

	@staticmethod
	def gen_replay(replay_buffer):
		while True:
			yield replay_buffer.sample()

	def can_replay(self):
		return self.num_added >= self.replay_starts

	def replay(self, replay_size=None):
		if not self.can_replay():
			return None
		if replay_size is None:
			replay_size = self.replay_batch_size

		with self.replay_timer:
			samples = {}
			transitions_count = 0
			for policy_id, replay_buffer in self.replay_buffers.items():
				batch_iter = self.gen_replay(replay_buffer) # generate new batches
				batch_iter = unique_everseen(batch_iter, key=lambda x:sorted(x['infos'][0]["batch_index"].items())) # skip duplicates
				batch_iter = islice(batch_iter, replay_size) # take the first n batches
				batch_list = list(batch_iter)
				samples[policy_id] = concat_batch = SampleBatch.concat_samples(batch_list) # add current batch
				transitions_count += concat_batch.count
			return MultiAgentBatch(samples, transitions_count)

	def update_priorities(self, prio_dict):
		with self.update_priorities_timer:
			for policy_id, new_batch in prio_dict.items():
				type_id = new_batch['infos'][0]["batch_type"]
				new_priority = self.replay_buffers[policy_id].get_batch_priority(new_batch)
				if isinstance(type_id,(tuple,list)):
					for sub_type_id in type_id:
						batch_index = new_batch['infos'][0]["batch_index"][sub_type_id]
						self.replay_buffers[policy_id].update_priority(new_priority, batch_index, sub_type_id)
				else:
					batch_index = new_batch['infos'][0]["batch_index"][type_id]
					self.replay_buffers[policy_id].update_priority(new_priority, batch_index, type_id)

	def stats(self, debug=False):
		stat = {
			"add_batch_time_ms": round(1000 * self.add_batch_timer.mean, 3),
			"replay_time_ms": round(1000 * self.replay_timer.mean, 3),
			"update_priorities_time_ms": round(1000 * self.update_priorities_timer.mean, 3),
		}
		for policy_id, replay_buffer in self.replay_buffers.items():
			stat.update({
				"policy_{}".format(policy_id): replay_buffer.stats(debug=debug)
			})
		return stat
