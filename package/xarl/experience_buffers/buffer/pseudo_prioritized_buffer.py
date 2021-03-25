# -*- coding: utf-8 -*-
import logging
import random
import numpy as np
import time
from xarl.experience_buffers.buffer.buffer import Buffer
from xarl.utils.segment_tree import SumSegmentTree, MinSegmentTree, MaxSegmentTree
import copy
import uuid
from xarl.utils.running_statistics import RunningStats

logger = logging.getLogger(__name__)

get_batch_infos = lambda x: x["infos"][0]
get_batch_indexes = lambda x: get_batch_infos(x)['batch_index']
get_batch_uid = lambda x: get_batch_infos(x)['batch_uid']

class PseudoPrioritizedBuffer(Buffer):
	
	def __init__(self, 
		priority_id,
		priority_aggregation_fn,
		cluster_size=None, 
		global_size=50000, 
		prioritization_alpha=0.6, 
		prioritization_importance_beta=0.4, 
		prioritization_importance_eta=1e-2,
		prioritization_epsilon=1e-6,
		prioritized_drop_probability=0.5, 
		global_distribution_matching=False, 
		cluster_prioritisation_strategy='highest',
		cluster_level_weighting=True,
		min_cluster_size_proportion=0.5,
		priority_lower_limit=None,
		max_age_window=None,
		seed=None,
	): # O(1)
		assert not prioritization_importance_beta or prioritization_importance_beta > 0., f"prioritization_importance_beta must be > 0, but it is {prioritization_importance_beta}"
		assert not prioritization_importance_eta or prioritization_importance_eta > 0, f"prioritization_importance_eta must be > 0, but it is {prioritization_importance_eta}"
		assert min_cluster_size_proportion >= 0, f"min_cluster_size_proportion must be >= 0, but it is {min_cluster_size_proportion}"
		self._priority_id = priority_id
		self._priority_lower_limit = priority_lower_limit
		self._priority_can_be_negative = priority_lower_limit is None or priority_lower_limit < 0
		self._priority_aggregation_fn = eval(priority_aggregation_fn) if self._priority_can_be_negative else (lambda x: eval(priority_aggregation_fn)(np.abs(x)))
		self._prioritization_alpha = prioritization_alpha # How much prioritization is used (0 - no prioritization, 1 - full prioritization)
		self._prioritization_importance_beta = prioritization_importance_beta # To what degree to use importance weights (0 - no corrections, 1 - full correction).
		self._prioritization_importance_eta = prioritization_importance_eta # Eta is a value > 0 that enables eta-weighting, thus allowing for importance weighting with priorities lower than 0. Eta is used to avoid importance weights equal to 0 when the sampled batch is the one with the highest priority. The closer eta is to 0, the closer to 0 would be the importance weight of the highest-priority batch.
		self._prioritization_epsilon = prioritization_epsilon # prioritization_epsilon to add to the priorities when updating priorities.
		self._prioritized_drop_probability = prioritized_drop_probability # remove the worst batch with this probability otherwise remove the oldest one
		self._global_distribution_matching = global_distribution_matching
		self._cluster_prioritisation_strategy = cluster_prioritisation_strategy
		self._cluster_level_weighting = cluster_level_weighting
		self._min_cluster_size_proportion = min_cluster_size_proportion
		self._weight_importance_by_update_time = self._max_age_window = max_age_window
		super().__init__(cluster_size=cluster_size, global_size=global_size, seed=seed)
		self._it_capacity = 1
		while self._it_capacity < self.cluster_size:
			self._it_capacity *= 2
		# self.priority_stats = RunningStats(window_size=self.global_size)
		self._base_time = time.time()
		self.min_cluster_size = 0
		self.max_cluster_size = self.cluster_size

	def is_weighting_expected_values(self):
		return self._prioritization_importance_beta
		
	def set(self, buffer): # O(1)
		assert isinstance(buffer, PseudoPrioritizedBuffer)
		super().set(buffer)
	
	def clean(self): # O(1)
		super().clean()
		self._sample_priority_tree = []
		if self._prioritized_drop_probability > 0:
			self._drop_priority_tree = []
		if self._prioritized_drop_probability < 1:
			self._insertion_time_tree = []
		if self._weight_importance_by_update_time:
			self._update_times = []
			
	def _add_type_if_not_exist(self, type_id): # O(1)
		if type_id not in self.types: # check it to avoid double insertion
			self.types[type_id] = type_ = len(self.type_keys)
			self.type_values.append(type_)
			self.type_keys.append(type_id)
			self.batches.append([])
			new_sample_priority_tree = SumSegmentTree(
				self._it_capacity, 
				with_min_tree=self._prioritization_importance_beta or self._priority_can_be_negative or (self._prioritized_drop_probability > 0 and not self._global_distribution_matching), 
				with_max_tree=self._priority_can_be_negative, 
			)
			self._sample_priority_tree.append(new_sample_priority_tree)
			if self._prioritized_drop_probability > 0:
				self._drop_priority_tree.append(
					MinSegmentTree(self._it_capacity,neutral_element=(float('inf'),-1))
					if self._global_distribution_matching else
					new_sample_priority_tree.min_tree
				)
			if self._prioritized_drop_probability < 1:
				self._insertion_time_tree.append(MinSegmentTree(self._it_capacity,neutral_element=(float('inf'),-1)))
			if self._weight_importance_by_update_time:
				self._update_times.append([])
		else:
			type_ = self.get_type(type_id)
		#################################################
		if self._sample_priority_tree[type_].inserted_elements == 0:
			self._sample_priority_tree[type_][0] = self._prioritization_epsilon # Inserting placeholder so that get_available_clusters returns the correct list
			logger.warning(f'Added a new cluster with id {type_id}, now there are {len(self.get_available_clusters())} different clusters.')
			self.resize_buffer()
			return True
		return False

	def resize_buffer(self):
		# print(random.random())
		new_max_cluster_size = self.get_max_cluster_size()
		if new_max_cluster_size == self.max_cluster_size:
			return
		# new_max_cluster_capacity = 1
		# while new_max_cluster_capacity < new_max_cluster_size:
		# 	new_max_cluster_capacity *= 2
		for t in self.type_values:
			elements_to_remove = max(0, self.count(t)-new_max_cluster_size)
			for _ in range(elements_to_remove):
				self.remove_batch(t, self.get_less_important_batch(t))
			# if self._prioritized_drop_probability > 0 and self._global_distribution_matching:
			# 	self._drop_priority_tree[t].resize(new_max_cluster_capacity)
			# if self._prioritized_drop_probability < 1:
			# 	self._insertion_time_tree[t].resize(new_max_cluster_capacity)
			# self._sample_priority_tree[t].resize(new_max_cluster_capacity)
		self.min_cluster_size = self.get_min_cluster_size()
		self.max_cluster_size = new_max_cluster_size
	
	def normalize_priority(self, priority): # O(1)
		# always add self._prioritization_epsilon so that there is no priority equal to the neutral value of a SumSegmentTree
		return (-1 if priority < 0 else 1)*(np.absolute(priority) + self._prioritization_epsilon)**self._prioritization_alpha

	def get_priority(self, idx, type_id):
		type_ = self.get_type(type_id)
		return self._sample_priority_tree[type_][idx]

	def remove_batch(self, type_, idx): # O(log)
		last_idx = len(self.batches[type_])-1
		assert idx <= last_idx, 'idx cannot be greater than last_idx'
		type_id = self.type_keys[type_]
		del get_batch_indexes(self.batches[type_][idx])[type_id]
		if idx == last_idx: # idx is the last, remove it
			if self._prioritized_drop_probability > 0 and self._global_distribution_matching:
				self._drop_priority_tree[type_][idx] = None # O(log)
			if self._prioritized_drop_probability < 1:
				self._insertion_time_tree[type_][idx] = None # O(log)
			if self._weight_importance_by_update_time:
				self._update_times[type_].pop()
			self._sample_priority_tree[type_][idx] = None # O(log)
			self.batches[type_].pop()
		elif idx < last_idx: # swap idx with the last element and then remove it
			if self._prioritized_drop_probability > 0 and self._global_distribution_matching:
				self._drop_priority_tree[type_][idx] = (self._drop_priority_tree[type_][last_idx][0],idx) # O(log)
				self._drop_priority_tree[type_][last_idx] = None # O(log)
			if self._prioritized_drop_probability < 1:
				self._insertion_time_tree[type_][idx] = (self._insertion_time_tree[type_][last_idx][0],idx) # O(log)
				self._insertion_time_tree[type_][last_idx] = None # O(log)
			if self._weight_importance_by_update_time:
				self._update_times[type_][idx] = self._update_times[type_].pop()
			self._sample_priority_tree[type_][idx] = self._sample_priority_tree[type_][last_idx] # O(log)
			self._sample_priority_tree[type_][last_idx] = None # O(log)
			batch = self.batches[type_][idx] = self.batches[type_].pop()
			get_batch_indexes(batch)[type_id] = idx

	def count(self, type_=None):
		if type_ is None:
			if len(self.batches) == 0:
				return 0
			return sum(t.inserted_elements for t in self._sample_priority_tree)
		return self._sample_priority_tree[type_].inserted_elements

	def get_available_clusters(self):
		return [x for x in self.type_values if not self.is_empty(x)]

	def get_min_cluster_size(self):
		return int(np.floor(self.global_size/(len(self.get_available_clusters())+self._min_cluster_size_proportion)))

	def get_avg_cluster_size(self):
		return int(np.floor(self.global_size/len(self.type_values)))

	def get_max_cluster_size(self):
		return int(np.ceil(self.get_min_cluster_size()*(1+self._min_cluster_size_proportion)))

	def get_cluster_capacity(self, segment_tree):
		return segment_tree.inserted_elements/self.max_cluster_size

	def get_relative_cluster_capacity(self, segment_tree):
		return segment_tree.inserted_elements/max(map(self.count, self.type_values))

	def get_cluster_priority(self, segment_tree, min_priority=0, avg_priority=None):
		if min_priority == avg_priority:
			return 0
		if segment_tree.inserted_elements == 0:
			return 0
		avg_cluster_priority = (segment_tree.sum()/segment_tree.inserted_elements) - min_priority # O(log)
		if avg_priority is not None:
			avg_cluster_priority = avg_cluster_priority/(avg_priority - min_priority) # avg_priority >= min_priority # scale by the global average priority
		assert avg_cluster_priority >= 0, f"avg_cluster_priority is {avg_cluster_priority}, it should be >= 0 otherwise the formula is wrong"
		return self.get_cluster_capacity(segment_tree)*avg_cluster_priority

	def get_cluster_capacity_dict(self):
		return dict(map(
			lambda x: (str(self.type_keys[x[0]]), self.get_cluster_capacity(x[1])), 
			enumerate(self._sample_priority_tree)
		))

	def get_cluster_priority_dict(self):
		min_priority = min(map(lambda x: x.min_tree.min()[0], self._sample_priority_tree)) # O(log)
		# avg_priority = sum(map(lambda x: x.sum(), self._sample_priority_tree))/sum(map(lambda x: x.inserted_elements, self._sample_priority_tree)) # O(log)
		return dict(map(
			# lambda x: (str(self.type_keys[x[0]]), self.get_cluster_priority(x[1], min_priority, avg_priority)), 
			lambda x: (str(self.type_keys[x[0]]), self.get_cluster_priority(x[1], min_priority)), 
			enumerate(self._sample_priority_tree)
		))

	def get_less_important_batch(self, type_):
		ptree = self._drop_priority_tree[type_] if random.random() <= self._prioritized_drop_probability else self._insertion_time_tree[type_]
		_,idx = ptree.min() # O(log)
		return idx

	def remove_less_important_batches(self, n):
		# Pick the right tree list
		if random.random() <= self._prioritized_drop_probability: 
			# Remove the batch with lowest priority
			tree_list = self._drop_priority_tree
		else: 
			# Remove the oldest batch
			tree_list = self._insertion_time_tree
		# Build the generator of the less important batch in every cluster
		# For all cluster to have the same size Y, we have that Y = N/C.
		# If we want to guarantee that every cluster contains at least pY elements while still reaching the maximum capacity of the whole buffer, then pY is the minimum size of a cluster.
		# If we want to constrain the maximum size of a cluster, we have to constrain with q the remaining (1-p)YC = (1-p)N elements so that (1-p)N = qpY, having that the size of a cluster is in [pY, pY+qpY].
		# Hence (1-p)N = qpN/C, then 1-p = qp/C, then p = 1/(1+q/C) = C/(C+q).
		# Therefore, we have that the minimum cluster's size pY = N/(C+q).
		less_important_batch_gen = (
			(*tree_list[type_].min(), type_) # O(log)
			for type_ in filter(lambda x: self.has_atleast(self.min_cluster_size, x), self.type_values)
			# for type_ in self.type_values
			# if not self.is_empty(type_)
		)
		less_important_batch_gen_len = len(self.type_values)
		# Remove the first N less important batches
		assert less_important_batch_gen_len > 0, "Cannot remove any batch from this buffer, it has too few elements"
		if n > 1 and less_important_batch_gen_len > 1:
			batches_to_remove = sorted(less_important_batch_gen, key=lambda x: x[0])
			n = min(n, len(batches_to_remove))
			for i in range(n):
				_, idx, type_ = batches_to_remove[i]
				self.remove_batch(type_, idx)
		else:
			_, idx, type_ = min(less_important_batch_gen, key=lambda x: x[0])
			self.remove_batch(type_, idx)
		if len(self.batches[type_]) == 0:
			logger.warning(f'Removed an old cluster with id {self.type_keys[type_]}, now there are {len(self.get_available_clusters())} different clusters.')
			self.resize_buffer()

	def _is_full_cluster(self, type_):
		return self.has_atleast(min(self.cluster_size,self.max_cluster_size), type_)
		
	def add(self, batch, type_id=0, update_prioritisation_weights=False): # O(log)
		self._add_type_if_not_exist(type_id)
		type_ = self.get_type(type_id)
		type_batch = self.batches[type_]
		idx = None
		if self._is_full_cluster(type_): # full cluster, remove from it
			idx = self.get_less_important_batch(type_)
		elif self.is_full_buffer(): # full buffer but not full cluster, remove the less important batch in the whole buffer
			self.remove_less_important_batches(1)
		if idx is None: # add new element to buffer
			idx = len(type_batch)
			type_batch.append(batch)
			if self._weight_importance_by_update_time:
				self._update_times[type_].append(self.timesteps)
		else:
			del get_batch_indexes(type_batch[idx])[type_id]
			type_batch[idx] = batch
		batch_infos = get_batch_infos(batch)
		if 'batch_index' not in batch_infos:
			batch_infos['batch_index'] = {}
		batch_infos['batch_index'][type_id] = idx
		batch_infos['batch_uid'] = str(uuid.uuid4()) # random unique id
		# Set insertion time
		if self._prioritized_drop_probability < 1:
			self._insertion_time_tree[type_][idx] = (self.get_relative_time(), idx) # O(log)
		# Set drop priority
		if self._prioritized_drop_probability > 0 and self._global_distribution_matching:
			self._drop_priority_tree[type_][idx] = (random.random(), idx) # O(log)
		# Set priority
		self.update_priority(batch, idx, type_id) # add batch
		if self._prioritization_importance_beta:
			if update_prioritisation_weights: # Update weights after updating priority
				self._cache_priorities()
				self.update_beta_weights(batch, idx, type_)
			elif 'weights' not in batch: # Add default weights
				batch['weights'] = np.ones(batch.count, dtype=np.float32)
		if self.global_size:
			assert self.count() <= self.global_size, 'Memory leak in replay buffer; v1'
			assert super().count() <= self.global_size, 'Memory leak in replay buffer; v2'
		return idx, type_id

	def _cache_priorities(self):
		if self._prioritization_importance_beta or self._cluster_prioritisation_strategy is not None:
			self.__min_priority_list = tuple(map(lambda x: x.min_tree.min()[0], self._sample_priority_tree)) # O(log)
			self.__min_priority = min(self.__min_priority_list)
		if self._prioritization_importance_beta and self._priority_lower_limit is None:
			self.__max_priority_list = tuple(map(lambda x: x.max_tree.max()[0], self._sample_priority_tree)) # O(log)
			self.__max_priority = max(self.__max_priority_list)
		if self._cluster_prioritisation_strategy is not None:
			# self.__avg_priority = sum(map(lambda x: x.sum(), self._sample_priority_tree))/sum(map(lambda x: x.inserted_elements, self._sample_priority_tree)) # O(log)
			# self.__cluster_priority_list = tuple(map(lambda x: self.get_cluster_priority(x, self.__min_priority, self.__avg_priority), self._sample_priority_tree)) # always > 0
			self.__cluster_priority_list = tuple(map(lambda x: self.get_cluster_priority(x, self.__min_priority if self._priority_lower_limit is None else self._priority_lower_limit), self._sample_priority_tree)) # always > 0
			# eta_normalise = lambda x: self.eta_normalisation(x, np.min(x), np.max(x), np.abs(np.std(x)/np.mean(x))) # using the coefficient of variation as eta
			# self.__cluster_priority_list = eta_normalise(eta_normalise(self.__cluster_priority_list)) # first eta-normalisation makes priorities in (0,1], but it inverts their magnitude # second eta-normalisation guarantees original priorities magnitude is preserved
			self.__min_cluster_priority = min(self.__cluster_priority_list)

	def sample_cluster(self):
		if self._cluster_prioritisation_strategy is not None:
			# assert self.__cluster_priority_list==tuple(map(lambda x: self.get_cluster_priority(x, self.__min_priority, self.__avg_priority), self._sample_priority_tree)), "Wrong clusters' prioritised sampling"
			type_cumsum = np.cumsum(self.__cluster_priority_list) # O(|self.type_keys|)
			type_mass = random.random() * type_cumsum[-1] # O(1)
			assert 0 <= type_mass, f'type_mass {type_mass} should be greater than 0'
			assert type_mass <= type_cumsum[-1], f'type_mass {type_mass} should be lower than {type_cumsum[-1]}'
			type_,_ = next(filter(lambda x: x[-1] >= type_mass and not self.is_empty(x[0]), enumerate(type_cumsum))) # O(|self.type_keys|)
		else:
			type_ = random.choice(tuple(filter(lambda x: not self.is_empty(x), self.type_values)))
		type_id = self.type_keys[type_]
		return type_id, type_

	def sample(self, n=1, recompute_priorities=True): # O(log)
		if recompute_priorities:
			self._cache_priorities()
		type_id, type_ = self.sample_cluster()
		cluster_sum_tree = self._sample_priority_tree[type_]
		type_batch = self.batches[type_]
		idx_list = [
			cluster_sum_tree.find_prefixsum_idx(prefixsum_fn=lambda mass: mass*random.random(), check_min=self._priority_can_be_negative) # O(log)
			for _ in range(n)
		]
		batch_list = [
			type_batch[idx] # O(1)
			for idx in idx_list
		]
		# Update weights
		if self._prioritization_importance_beta: # Update weights
			for batch,idx in zip(batch_list,idx_list):
				self.update_beta_weights(batch, idx, type_)
		return batch_list

	@staticmethod
	def eta_normalisation(priorities, min_priority, max_priority, eta):
		priorities = np.clip(priorities, min_priority, max_priority)
		upper_max_priority = max_priority*((1+eta) if max_priority >= 0 else (1-eta))
		if upper_max_priority == min_priority: 
			return 1.
		assert upper_max_priority > min_priority, f"upper_max_priority must be > min_priority, but it is {upper_max_priority} while min_priority is {min_priority}"
		return (upper_max_priority - priorities)/(upper_max_priority - min_priority) # in (0,1]: the closer is cluster_sum_tree[idx] to max_priority, the lower is the weight

	def update_beta_weights(self, batch, idx, type_):
		##########
		# Get priority weight
		batch_priority = self._sample_priority_tree[type_][idx]
		min_priority = self.__min_priority_list[type_] if self._cluster_level_weighting else self.__min_priority
		# assert self.__min_priority_list == tuple(map(lambda x: x.min_tree.min()[0], self._sample_priority_tree)), "Wrong beta updates"
		if self._priority_lower_limit is None: # We still need to prevent over-fitting on most frequent batches: https://datascience.stackexchange.com/questions/32873/prioritized-replay-what-does-importance-sampling-really-do
			max_priority = self.__max_priority_list[type_] if self._cluster_level_weighting else self.__max_priority
			weight = self.eta_normalisation(batch_priority, min_priority, max_priority, self._prioritization_importance_eta)
			if self._cluster_level_weighting and self._cluster_prioritisation_strategy is not None and self.__cluster_priority_list[type_] != 0:
				weight *= self.__min_cluster_priority / self.__cluster_priority_list[type_]
		else:
			assert min_priority > self._priority_lower_limit, f"min_priority must be > priority_lower_limit, if beta is not None and priority_can_be_negative is False, but it is {min_priority}"
			batch_priority = np.maximum(batch_priority, min_priority) # no need for this instruction if we are not averaging/maxing clusters' min priorities
			weight = (min_priority - self._priority_lower_limit) / (batch_priority - self._priority_lower_limit) # default, not compatible with negative priorities # in (0,1]: the closer is cluster_sum_tree[idx] to max_priority, the lower is the weight
			if self._cluster_level_weighting and self._cluster_prioritisation_strategy is not None and self.__cluster_priority_list[type_] != 0:
				weight *= self.__min_cluster_priority / self.__cluster_priority_list[type_]
		weight = weight**self._prioritization_importance_beta
		##########
		# Add age weight
		if self._weight_importance_by_update_time:
			relative_age = self.timesteps - self._update_times[type_][idx]
			# if relative_age > self._max_age_window:
			# 	weight *= 1/self._max_age_window
			age_weight = max(1,(self._max_age_window - relative_age))/self._max_age_window
			weight *= age_weight # batches with outdated priorities should have a lower weight, they might be just noise
		##########
		batch['weights'] = np.full(batch.count, weight, dtype=np.float32)

	def get_batch_priority(self, batch):
		return self._priority_aggregation_fn(batch[self._priority_id])
	
	def update_priority(self, new_batch, idx, type_id=0): # O(log)
		type_ = self.get_type(type_id)
		if idx >= len(self.batches[type_]):
			return
		if get_batch_uid(new_batch) != get_batch_uid(self.batches[type_][idx]):
			return
		# for k,v in self.batches[type_][idx].data.items():
		# 	if not np.array_equal(new_batch[k],v):
		# 		print(k,v,new_batch[k])
		new_priority = self.get_batch_priority(new_batch)
		normalized_priority = self.normalize_priority(new_priority)
		# self.priority_stats.push(normalized_priority)
		# Update priority
		self._sample_priority_tree[type_][idx] = normalized_priority # O(log)
		if self._weight_importance_by_update_time:
			self._update_times[type_][idx] = self.timesteps # O(1)

	def get_relative_time(self):
		return time.time()-self._base_time

	def stats(self, debug=False):
		stats_dict = super().stats(debug)
		stats_dict.update({
			'cluster_capacity':self.get_cluster_capacity_dict(),
			'cluster_priority': self.get_cluster_priority_dict(),
		})
		return stats_dict
