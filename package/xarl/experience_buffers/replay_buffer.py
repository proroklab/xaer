import collections
import logging
import numpy as np
import platform
from more_itertools import unique_everseen
from itertools import islice
import copy
import threading 
import random

# Import ray before psutil will make sure we use psutil's bundled version
import ray  # noqa F401
import psutil  # noqa E402

from xarl.experience_buffers.buffer.pseudo_prioritized_buffer import PseudoPrioritizedBuffer, get_batch_infos, get_batch_indexes, get_batch_uid
from xarl.experience_buffers.buffer.buffer import Buffer
from xarl.utils import ReadWriteLock

from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch, DEFAULT_POLICY_ID
from ray.util.iter import ParallelIteratorWorker
from ray.rllib.utils.timer import TimerStat

logger = logging.getLogger(__name__)

def apply_to_batch_once(fn, batch_list):
	updated_batch_dict = {
		get_batch_uid(x): fn(x) 
		for x in unique_everseen(batch_list, key=get_batch_uid)
	}
	return list(map(lambda x: updated_batch_dict[get_batch_uid(x)], batch_list))

class LocalReplayBuffer(ParallelIteratorWorker):
	"""A replay buffer shard.

	Ray actors are single-threaded, so for scalability multiple replay actors
	may be created to increase parallelism."""

	def __init__(self, 
		prioritized_replay=True,
		buffer_options=None, 
		learning_starts=1000, 
	):
		self.prioritized_replay = prioritized_replay
		self.buffer_options = {} if not buffer_options else buffer_options
		self.replay_starts = learning_starts
		self._buffer_lock = ReadWriteLock() 

		ParallelIteratorWorker.__init__(self, None, False)

		def new_buffer():
			return PseudoPrioritizedBuffer(**self.buffer_options) if self.prioritized_replay else Buffer(**self.buffer_options)

		self.replay_buffers = collections.defaultdict(new_buffer)

		# Metrics
		self.add_batch_timer = TimerStat()
		self.replay_timer = TimerStat()
		self.update_priorities_timer = TimerStat()
		self.num_added = 0

	def add_batch(self, batch, on_policy=False):
		# Make a copy so the replay buffer doesn't pin plasma memory.
		batch = batch.copy()
		batch['infos'] = copy.deepcopy(batch['infos'])
		# Get batch's type
		batch_type = get_batch_infos(batch)["batch_type"]
		has_multiple_types = isinstance(batch_type,(tuple,list))
		# Handle everything as if multiagent
		if isinstance(batch, SampleBatch):
			batch = MultiAgentBatch({DEFAULT_POLICY_ID: batch}, batch.count)
		with self.add_batch_timer:
			self.num_added += len(batch.policy_batches)
			random_type_id = random.choice(batch_type) if has_multiple_types else batch_type
			self._buffer_lock.acquire_write()
			for policy_id, b in batch.policy_batches.items():
				# If has_multiple_types is True: no need for duplicating the batch across multiple clusters, just insert into one of them, randomly. It is a prioritised buffer, clusters will be fairly represented, with minimum overhead.
				self.replay_buffers[policy_id].add(batch=b, type_id=random_type_id, on_policy=on_policy)
			self._buffer_lock.release_write()
		return batch

	def can_replay(self):
		return self.num_added >= self.replay_starts

	def replay(self, batch_count=1, cluster_overview_size=None, update_replayed_fn=None):
		if not self.can_replay():
			return None
		if not cluster_overview_size:
			cluster_overview_size = batch_count
		else:
			cluster_overview_size = min(cluster_overview_size,batch_count)

		with self.replay_timer:
			batch_list = [{} for _ in range(batch_count)]
			for policy_id, replay_buffer in self.replay_buffers.items():
				if replay_buffer.is_empty():
					continue
				# batch_iter = replay_buffer.sample(batch_count)
				batch_size_list = [cluster_overview_size]*(batch_count//cluster_overview_size)
				if batch_count%cluster_overview_size > 0:
					batch_size_list.append(batch_count%cluster_overview_size)
				self._buffer_lock.acquire_read()
				batch_iter = sum(map(replay_buffer.sample,batch_size_list), [])
				self._buffer_lock.release_read()
				if update_replayed_fn:
					batch_iter = apply_to_batch_once(update_replayed_fn, batch_iter)
				for i,batch in enumerate(batch_iter):
					batch_list[i][policy_id] = batch
		return (
			MultiAgentBatch(samples, max(map(lambda x:x.count, samples.values())))
			for samples in batch_list
		)

	def replay_n_concatenate(self, batch_count=1, cluster_overview_size=None, update_replayed_fn=None):
		if not self.can_replay():
			return None
		if not cluster_overview_size:
			cluster_overview_size = batch_count
		else:
			cluster_overview_size = min(cluster_overview_size,batch_count)

		with self.replay_timer:
			samples = {}
			for policy_id, replay_buffer in self.replay_buffers.items():
				# batch_iter = replay_buffer.sample(batch_count)
				batch_size_list = [cluster_overview_size]*(batch_count//cluster_overview_size)
				if batch_count%cluster_overview_size > 0:
					batch_size_list.append(batch_count%cluster_overview_size)
				self._buffer_lock.acquire_read()
				batch_iter = sum(map(replay_buffer.sample,batch_size_list), [])
				self._buffer_lock.release_read()
				if update_replayed_fn:
					batch_iter = apply_to_batch_once(update_replayed_fn, batch_iter)
				samples[policy_id] = SampleBatch.concat_samples(batch_iter)
			return MultiAgentBatch(samples, max(map(lambda x:x.count, samples.values())))

	def update_priorities(self, prio_dict):
		if not self.prioritized_replay:
			return
		with self.update_priorities_timer:
			self._buffer_lock.acquire_write()
			for policy_id, new_batch in prio_dict.items():
				for type_id,batch_index in get_batch_indexes(new_batch).items():
					self.replay_buffers[policy_id].update_priority(new_batch, batch_index, type_id)
			self._buffer_lock.release_write()

	def stats(self, debug=False):
		stat = {
			"add_batch_time_ms": round(1000 * self.add_batch_timer.mean, 3),
			"replay_time_ms": round(1000 * self.replay_timer.mean, 3),
			"update_priorities_time_ms": round(1000 * self.update_priorities_timer.mean, 3),
		}
		for policy_id, replay_buffer in self.replay_buffers.items():
			stat.update({
				policy_id: replay_buffer.stats(debug=debug)
			})
		return stat
