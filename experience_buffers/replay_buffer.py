import collections
import logging
import numpy as np
import platform

# Import ray before psutil will make sure we use psutil's bundled version
import ray  # noqa F401
import psutil  # noqa E402

from experience_buffers.buffer.pseudo_prioritized_buffer import PseudoPrioritizedBuffer
from experience_buffers.buffer.buffer import Buffer

from ray.rllib.policy.sample_batch import SampleBatch
from ray.util.iter import ParallelIteratorWorker
from ray.rllib.utils.timer import TimerStat

# Constant that represents all policies in lockstep replay mode.
_ALL_POLICIES = "__all__"

logger = logging.getLogger(__name__)

class LocalReplayBuffer(ParallelIteratorWorker):
	"""A replay buffer shard.

	Ray actors are single-threaded, so for scalability multiple replay actors
	may be created to increase parallelism."""

	def __init__(self, 
		prioritized_replay=True,
		buffer_options=None, 
		learning_starts=1000, 
		replay_sequence_length=1, 
	):
		self.prioritized_replay = prioritized_replay
		self.buffer_options = {} if not buffer_options else buffer_options
		self.replay_starts = learning_starts
		self.replay_sequence_length = replay_sequence_length

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
		batch_type = batch["batch_types"][0]
		with self.add_batch_timer:
			self.replay_buffers[_ALL_POLICIES].add(batch, batch_type)
		self.num_added += batch.count
		return batch

	def replay(self):
		if self.num_added < self.replay_starts:
			return None

		with self.replay_timer:
			return self.replay_buffers[_ALL_POLICIES].sample()

	def update_priority(self, new_batch):
		if not self.prioritized_replay:
			return
		batch_index = new_batch["batch_indexes"][0]
		type_id = new_batch["batch_types"][0]
		new_priority = self.replay_buffers[_ALL_POLICIES].get_batch_priority(new_batch)
		
		with self.update_priorities_timer:
			# old_p = self.replay_buffers[_ALL_POLICIES].get_priority(batch_index, type_id)
			self.replay_buffers[_ALL_POLICIES].update_priority(new_priority, batch_index, type_id)
			# new_p = self.replay_buffers[_ALL_POLICIES].get_priority(batch_index, type_id)
			# print(old_p,new_p)

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

ReplayActor = ray.remote(num_cpus=0)(LocalReplayBuffer)
