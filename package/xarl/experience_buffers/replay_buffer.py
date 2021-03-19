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

class SimpleReplayBuffer:
	"""Simple replay buffer that operates over batches."""

	def __init__(self, num_slots):
		"""Initialize SimpleReplayBuffer.

		Args:
			num_slots (int): Number of batches to store in total.
		"""
		self.num_slots = num_slots
		self.replay_batches = []
		self.replay_index = 0

	def can_replay(self):
		return len(self.replay_batches) >= self.num_slots

	def add_batch(self, sample_batch):
		if self.num_slots > 0:
			if len(self.replay_batches) < self.num_slots:
				self.replay_batches.append(sample_batch)
			else:
				self.replay_batches[self.replay_index] = sample_batch
				self.replay_index = (self.replay_index+1)%self.num_slots

	def replay(self, batch_count=1):
		return random.sample(self.replay_batches, batch_count)

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
		dummy_buffer = PseudoPrioritizedBuffer(**self.buffer_options)
		self.buffer_size = dummy_buffer.global_size
		self.is_weighting_expected_values = dummy_buffer.is_weighting_expected_values()
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

	def add_batch(self, batch, update_prioritisation_weights=False):
		# Handle everything as if multiagent
		if isinstance(batch, SampleBatch):
			batch = MultiAgentBatch({DEFAULT_POLICY_ID: batch}, batch.count)
		self.num_added += len(batch.policy_batches)
		with self.add_batch_timer:
			self._buffer_lock.acquire_write()
			for policy_id, sub_batch in batch.policy_batches.items():
				batch_type = get_batch_infos(sub_batch)["batch_type"]
				if isinstance(batch_type,(tuple,list)):
					# # If has_multiple_types is True: no need for duplicating the batch across multiple clusters unless they are invalid, just insert into one of them, randomly. It is a prioritised buffer, clusters will be fairly represented, with minimum overhead.
					# sub_type_list = tuple(filter(lambda x: not self.replay_buffers[policy_id].is_valid_cluster(x), batch_type))
					# if len(sub_type_list) == 0:
					# 	sub_type_list = (random.choice(batch_type),)
					sub_type_list = (random.choice(batch_type),)
					# sub_type_list = batch_type
				else:
					sub_type_list = (batch_type,)
				for sub_type in sub_type_list: 
					# Make a deep copy so the replay buffer doesn't pin plasma memory.
					batch.policy_batches[policy_id] = sub_batch = sub_batch.copy()
					# Make a deep copy of infos so that for every sub_type the infos dictionary is different
					sub_batch['infos'] = copy.deepcopy(sub_batch['infos'])
					self.replay_buffers[policy_id].add(batch=sub_batch, type_id=sub_type, update_prioritisation_weights=update_prioritisation_weights)
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
				batch_iter = []
				for i,n in enumerate(batch_size_list):
					batch_iter += replay_buffer.sample(n,recompute_priorities=i==0)
				self._buffer_lock.release_read()
				if update_replayed_fn:
					self._buffer_lock.acquire_write()
					batch_iter = apply_to_batch_once(update_replayed_fn, batch_iter)
					self._buffer_lock.release_write()
				for i,batch in enumerate(batch_iter):
					batch_list[i][policy_id] = batch
		return (
			MultiAgentBatch(samples, max(map(lambda x:x.count, samples.values())))
			for samples in batch_list
		)

	def increase_train_steps(self, t=1):
		for replay_buffer in self.replay_buffers.values():
			replay_buffer.increase_steps(t)

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
