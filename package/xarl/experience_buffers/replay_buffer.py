import collections
import logging
import numpy as np
import platform
from more_itertools import unique_everseen
from itertools import islice
import copy
import threading 

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
		update_only_sampled_cluster=True,
	):
		self.update_only_sampled_cluster = update_only_sampled_cluster
		self.prioritized_replay = prioritized_replay
		self.buffer_options = {} if not buffer_options else buffer_options
		self.replay_starts = learning_starts
		# self._buffer_lock = threading.Lock() 

		ParallelIteratorWorker.__init__(self, None, False)

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
		batch['infos'] = copy.deepcopy(batch['infos'])
		batch_type = batch['infos'][0]["batch_type"]
		# Handle everything as if multiagent
		if isinstance(batch, SampleBatch):
			batch = MultiAgentBatch({DEFAULT_POLICY_ID: batch}, batch.count)
		with self.add_batch_timer:
			# with self._buffer_lock:
			for policy_id, b in batch.policy_batches.items():
				self.num_added += 1
				if isinstance(batch_type,(tuple,list)):
					for sub_batch_type in batch_type:
						if self.update_only_sampled_cluster:
							b_copy = b.copy()
							b_copy['infos'] = copy.deepcopy(b_copy['infos'])
							self.replay_buffers[policy_id].add(b_copy, sub_batch_type)
						else:
							self.replay_buffers[policy_id].add(b, sub_batch_type)
				else:
					self.replay_buffers[policy_id].add(b, batch_type)
			return batch

	def can_replay(self):
		return self.num_added >= self.replay_starts

	def replay(self, replay_size=1, filter_duplicates=True):
		def gen_sample(replay_buffer):
			while True:
				yield replay_buffer.sample()
		if not self.can_replay():
			return None

		with self.replay_timer:
			# with self._buffer_lock:
			batch_list = [{} for _ in range(replay_size)]
			for policy_id, replay_buffer in self.replay_buffers.items():
				batch_iter = gen_sample(replay_buffer) # generate new batches
				if filter_duplicates:
					batch_iter = unique_everseen(batch_iter, key=lambda x:sorted(x['infos'][0]["batch_index"].items())) # skip duplicates
				batch_iter = islice(batch_iter, replay_size) # take the first n batches
				for i,batch in enumerate(batch_iter):
					batch_list[i][policy_id] = batch
		return [
			MultiAgentBatch(samples, max(map(lambda x:x.count, samples.values())))
			for samples in batch_list
		] 

	def update_priorities(self, prio_dict):
		with self.update_priorities_timer:
			# with self._buffer_lock:
			for policy_id, new_batch in prio_dict.items():
				# Make a copy so the replay buffer doesn't pin plasma memory.
				new_batch = new_batch.copy()
				new_batch['infos'] = copy.deepcopy(new_batch['infos'])
				for type_id,batch_index in new_batch['infos'][0]["batch_index"].items():
					self.replay_buffers[policy_id].update_priority(new_batch, batch_index, type_id)

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
