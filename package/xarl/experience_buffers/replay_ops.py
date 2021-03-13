from typing import List
import random
import numpy as np
from more_itertools import unique_everseen

from ray.util.iter import LocalIterator, _NextValueNotReady
from ray.util.iter_metrics import SharedMetrics
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch, DEFAULT_POLICY_ID
from ray.rllib.execution.learner_thread import LearnerThread, get_learner_stats
from ray.rllib.execution.multi_gpu_learner import TFMultiGPULearner, get_learner_stats as get_gpu_learner_stats

from xarl.experience_buffers.replay_buffer import SimpleReplayBuffer, LocalReplayBuffer, get_batch_infos
from xarl.experience_buffers.clustering_scheme import *

def get_clustered_replay_buffer(config):
	assert config["batch_mode"] == "complete_episodes" or not config["cluster_with_episode_type"], f"This algorithm requires 'complete_episodes' as batch_mode when 'cluster_with_episode_type' is True"
	local_replay_buffer = LocalReplayBuffer(
		prioritized_replay=config["prioritized_replay"],
		buffer_options=config["buffer_options"], 
		learning_starts=config["learning_starts"], 
	)
	clustering_scheme = eval(config["clustering_scheme"])()
	return local_replay_buffer, clustering_scheme

def assign_types(batch, clustering_scheme, batch_fragment_length, with_episode_type=True):
	if with_episode_type:
		batch_list = []
		for episode in batch.split_by_episode():
			sub_batch_list = episode.timeslices(batch_fragment_length) if episode.count > batch_fragment_length else [episode]
			episode_type = clustering_scheme.get_episode_type(sub_batch_list)
			for sub_batch in sub_batch_list:
				sub_batch_type = clustering_scheme.get_batch_type(sub_batch, episode_type)
				get_batch_infos(sub_batch)['batch_type'] = sub_batch_type
			batch_list += sub_batch_list
		return batch_list
	sub_batch_list = batch.timeslices(batch_fragment_length) if batch.count > batch_fragment_length else [batch]
	for sub_batch in sub_batch_list:
		sub_batch_type = clustering_scheme.get_batch_type(sub_batch)
		get_batch_infos(sub_batch)['batch_type'] = sub_batch_type
	return sub_batch_list

def get_update_replayed_batch_fn(local_replay_buffer, local_worker, postprocess_trajectory_fn):
	def update_replayed_fn(samples):
		if isinstance(samples, MultiAgentBatch):
			for pid, batch in samples.policy_batches.items():
				if pid not in local_worker.policies_to_train:
					continue
				policy = local_worker.get_policy(pid)
				samples.policy_batches[pid] = postprocess_trajectory_fn(policy, batch)
			local_replay_buffer.update_priorities(samples.policy_batches)
		else:
			samples = postprocess_trajectory_fn(local_worker.policy_map[DEFAULT_POLICY_ID], samples)
			local_replay_buffer.update_priorities({DEFAULT_POLICY_ID:samples})
		return samples
	return update_replayed_fn

def clean_batch(batch, keys_to_keep=None, keep_only_keys_to_keep=False):
	if isinstance(batch, MultiAgentBatch):
		for b in batch.policy_batches.values():
			for k,v in list(b.data.items()):
				if keys_to_keep and k in keys_to_keep:
					continue
				if keep_only_keys_to_keep or not isinstance(v, np.ndarray):
					del b.data[k]
	else:
		for k,v in list(batch.data.items()):
			if keys_to_keep and k in keys_to_keep:
				continue
			if keep_only_keys_to_keep or not isinstance(v, np.ndarray):
				del batch.data[k]
	return batch

def add_buffer_metrics(results, buffer):
	results['buffer']=buffer.stats()
	return results

class StoreToReplayBuffer:
	def __init__(self, local_buffer: LocalReplayBuffer = None):
		self.local_actor = local_buffer
		
	def __call__(self, batch: SampleBatchType):
		return self.local_actor.add_batch(batch)

def Replay(local_buffer, replay_batch_size=1, cluster_overview_size=None, update_replayed_fn=None):
	def gen_replay(_):
		while True:
			batch_list = local_buffer.replay(
				batch_count=replay_batch_size, 
				cluster_overview_size=cluster_overview_size,
				update_replayed_fn=update_replayed_fn,
			)
			if not batch_list:
				yield _NextValueNotReady()
			else:
				yield batch_list
	return LocalIterator(gen_replay, SharedMetrics())

class MixInReplay:
	"""This operator adds replay to a stream of experiences.

	It takes input batches, and returns a list of batches that include replayed
	data as well. The number of replayed batches is determined by the
	configured replay proportion. The max age of a batch is determined by the
	number of replay slots.
	"""

	def __init__(self, local_buffer, replay_proportion, cluster_overview_size=None, update_replayed_fn=None, sample_also_from_buffer_of_recent_elements=False):
		self.replay_buffer = local_buffer
		self.replay_proportion = replay_proportion
		self.update_replayed_fn = update_replayed_fn
		self.cluster_overview_size = cluster_overview_size
		self.buffer_of_recent_elements = SimpleReplayBuffer(local_buffer.buffer_size) if sample_also_from_buffer_of_recent_elements else None

	def __call__(self, sample_batch):
		# n = np.random.poisson(self.replay_proportion)
		n = int(self.replay_proportion//1)
		if self.replay_proportion%1 > 0 and random.random() <= self.replay_proportion%1:
			n += 1
		output_batches = []
		# Put sample_batch in the experience buffer and add it to the output_batches
		if isinstance(sample_batch, SampleBatch):
			sample_batch = MultiAgentBatch({DEFAULT_POLICY_ID: sample_batch}, sample_batch.count)
		output_batches.append(sample_batch)
		if self.buffer_of_recent_elements:
			self.buffer_of_recent_elements.add_batch(sample_batch)
		self.replay_buffer.add_batch(sample_batch) # Set update_prioritisation_weights=True for updating importance weights
		# Sample n batches from the buffer
		if self.replay_buffer.can_replay() and n > 0:
			if self.buffer_of_recent_elements and self.buffer_of_recent_elements.can_replay():
				n_of_old_elements = random.randint(0,n)
				if n_of_old_elements > 0:
					output_batches += self.replay_buffer.replay(
						batch_count=n_of_old_elements,
						cluster_overview_size=self.cluster_overview_size,
						update_replayed_fn=self.update_replayed_fn,
					)
				if n_of_old_elements != n:
					output_batches += self.buffer_of_recent_elements.replay(n-n_of_old_elements)
			else:
				output_batches += self.replay_buffer.replay(
					batch_count=n,
					cluster_overview_size=self.cluster_overview_size,
					update_replayed_fn=self.update_replayed_fn,
				)
		return output_batches

class BatchLearnerThread(LearnerThread):
	def step(self):
		with self.queue_timer:
			batch, _ = self.minibatch_buffer.get()

		with self.grad_timer:
			fetches = self.local_worker.learn_on_batch(batch)
			self.weights_updated = True
			self.stats = get_learner_stats(fetches)

		self.num_steps += 1
		self.outqueue.put((batch, self.stats))
		self.learner_queue_size.push(self.inqueue.qsize())

class BatchTFMultiGPULearner(TFMultiGPULearner):
	def step(self):
		assert self.loader_thread.is_alive()
		with self.load_wait_timer:
			opt, released = self.minibatch_buffer.get()

		with self.grad_timer:
			fetches = opt.optimize(self.sess, 0)
			self.weights_updated = True
			self.stats = get_gpu_learner_stats(fetches)

		if released:
			self.idle_optimizers.put(opt)

		self.outqueue.put((opt, self.stats))
		self.learner_queue_size.push(self.inqueue.qsize())

def xa_make_learner_thread(local_worker, config):
	if config["num_gpus"] > 1 or config["num_data_loader_buffers"] > 1:
		logger.info(
			"Enabling multi-GPU mode, {} GPUs, {} parallel loaders".format(
				config["num_gpus"], config["num_data_loader_buffers"]))
		if config["num_data_loader_buffers"] < config["minibatch_buffer_size"]:
			raise ValueError(
				"In multi-gpu mode you must have at least as many "
				"parallel data loader buffers as minibatch buffers: "
				"{} vs {}".format(config["num_data_loader_buffers"],
								  config["minibatch_buffer_size"]))
		learner_thread = TFMultiGPULearner(
			local_worker,
			num_gpus=config["num_gpus"],
			lr=config["lr"],
			train_batch_size=config["train_batch_size"],
			num_data_loader_buffers=config["num_data_loader_buffers"],
			minibatch_buffer_size=config["minibatch_buffer_size"],
			num_sgd_iter=config["num_sgd_iter"],
			learner_queue_size=config["learner_queue_size"],
			learner_queue_timeout=config["learner_queue_timeout"])
	else:
		learner_thread = BatchLearnerThread(
			local_worker,
			minibatch_buffer_size=config["minibatch_buffer_size"],
			num_sgd_iter=config["num_sgd_iter"],
			learner_queue_size=config["learner_queue_size"],
			learner_queue_timeout=config["learner_queue_timeout"])
	return learner_thread
