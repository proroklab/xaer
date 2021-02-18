# -*- coding: utf-8 -*-
import logging
from random import choice, random, randint
import numpy as np
import time
from xarl.experience_buffers.buffer.buffer import Buffer
from xarl.utils.segment_tree import SumSegmentTree, MinSegmentTree
import copy
import uuid
from xarl.utils.running_statistics import RunningStats

logger = logging.getLogger(__name__)

get_batch_infos = lambda x: x["infos"][0]
get_batch_indexes = lambda x: get_batch_infos(x)['batch_index']
get_batch_uid = lambda x: get_batch_infos(x)['batch_uid']

class PseudoPrioritizedBuffer(Buffer):
	# __slots__ = ('_priority_id','_priority_aggregation_fn','_alpha','_beta','_eta','_epsilon','_prioritized_drop_probability','_global_distribution_matching','_it_capacity','_sample_priority_tree','_drop_priority_tree','_insertion_time_tree','_cluster_prioritisation','_cluster_prioritisation_strategy','_cluster_level_weighting','_min_cluster_size_proportion','_priority_can_be_negative')
	
	def __init__(self, 
		priority_id,
		priority_aggregation_fn,
		cluster_size=None, 
		global_size=50000, 
		alpha=0.6, 
		beta=0.4, 
		eta=1e-2,
		epsilon=1e-6,
		prioritized_drop_probability=0.5, 
		global_distribution_matching=False, 
		cluster_prioritisation_strategy='highest',
		cluster_prioritisation_smoothing_degree=1, # The higher it is, the more frequently sampled will be the clusters with the lowest priority. This is how many standard deviations to consider for estimating the upper maximum priority in the eta-normalisation of clusters priorities.
		cluster_level_weighting=True,
		min_cluster_size_proportion=0.5,
		priority_can_be_negative=True,
	): # O(1)
		assert not beta or beta > 0., f"beta must be > 0, but it is {beta}"
		assert not eta or eta > 0, f"eta must be > 0, but it is {eta}"
		assert min_cluster_size_proportion >= 0, f"min_cluster_size_proportion must be >= 0, but it is {min_cluster_size_proportion}"
		self._priority_id = priority_id
		self._priority_can_be_negative = priority_can_be_negative
		self._priority_aggregation_fn = eval(priority_aggregation_fn) if priority_can_be_negative else (lambda x: eval(priority_aggregation_fn)(np.abs(x)))
		self._alpha = alpha # How much prioritization is used (0 - no prioritization, 1 - full prioritization)
		self._beta = beta # To what degree to use importance weights (0 - no corrections, 1 - full correction).
		self._eta = eta # Eta is a value > 0 that enables eta-weighting, thus allowing for importance weighting with priorities lower than 0. Eta is used to avoid importance weights equal to 0 when the sampled batch is the one with the highest priority. The closer eta is to 0, the closer to 0 would be the importance weight of the highest-priority batch.
		self._cluster_prioritisation_smoothing_degree = cluster_prioritisation_smoothing_degree
		self._epsilon = epsilon # Epsilon to add to the priorities when updating priorities.
		self._prioritized_drop_probability = prioritized_drop_probability # remove the worst batch with this probability otherwise remove the oldest one
		self._global_distribution_matching = global_distribution_matching
		self._cluster_prioritisation_strategy = cluster_prioritisation_strategy
		self._cluster_level_weighting = cluster_level_weighting
		self._min_cluster_size_proportion = min_cluster_size_proportion
		super().__init__(cluster_size=cluster_size, global_size=global_size)
		self._it_capacity = 1
		while self._it_capacity < self.cluster_size:
			self._it_capacity *= 2
		# self.priority_stats = RunningStats(window_size=self.global_size)
		
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
			
	def _add_type_if_not_exist(self, type_id): # O(1)
		if type_id in self.types: # check it to avoid double insertion
			return False
		self.types[type_id] = type_ = len(self.type_keys)
		self.type_values.append(type_)
		self.type_keys.append(type_id)
		self.batches.append([])
		new_sample_priority_tree = SumSegmentTree(
			self._it_capacity, 
			with_min_tree=self._beta or self._priority_can_be_negative or (self._prioritized_drop_probability > 0 and not self._global_distribution_matching), 
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
		logger.warning(f'Added a new cluster with id {type_id}, now there are {len(self.type_values)} different clusters.')
		new_max_cluster_size = self.get_max_cluster_size()
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
		return True
	
	def normalize_priority(self, priority): # O(1)
		# always add self._epsilon so that there is no priority equal to the neutral value of a SumSegmentTree
		return (-1 if priority < 0 else 1)*(np.absolute(priority) + self._epsilon)**self._alpha

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
			self._sample_priority_tree[type_][idx] = None # O(log)
			self.batches[type_].pop()
		elif idx < last_idx: # swap idx with the last element and then remove it
			if self._prioritized_drop_probability > 0 and self._global_distribution_matching:
				self._drop_priority_tree[type_][idx] = (self._drop_priority_tree[type_][last_idx][0],idx) # O(log)
				self._drop_priority_tree[type_][last_idx] = None # O(log)
			if self._prioritized_drop_probability < 1:
				self._insertion_time_tree[type_][idx] = (self._insertion_time_tree[type_][last_idx][0],idx) # O(log)
				self._insertion_time_tree[type_][last_idx] = None # O(log)
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

	def get_less_important_batch(self, type_):
		ptree = self._drop_priority_tree[type_] if random() <= self._prioritized_drop_probability else self._insertion_time_tree[type_]
		_,idx = ptree.min() # O(log)
		return idx

	def get_min_cluster_size(self):
		return int(np.floor(self.global_size/(len(self.type_values)+self._min_cluster_size_proportion)))

	def get_max_cluster_size(self):
		return int(np.floor(self.get_min_cluster_size()*(1+self._min_cluster_size_proportion)))

	def get_cluster_capacity(self, segment_tree):
		return segment_tree.inserted_elements/self.max_cluster_size

	def get_cluster_priority(self, segment_tree):
		avg_cluster_priority = segment_tree.sum()/segment_tree.inserted_elements # O(log)
		return self.get_cluster_capacity(segment_tree)*avg_cluster_priority # A min_cluster_size_proportion lower than 1 guarantees that, taking the sum instead of the average, the resulting type priority is still relying on the average clusters' priority

	def get_cluster_capacity_dict(self):
		return dict(map(
			lambda x: (str(self.type_keys[x[0]]), self.get_cluster_capacity(x[1])), 
			enumerate(self._sample_priority_tree)
		))

	def get_cluster_priority_dict(self):
		return dict(map(
			lambda x: (str(self.type_keys[x[0]]), self.get_cluster_priority(x[1])), 
			enumerate(self._sample_priority_tree)
		))

	def is_full_cluster(self, type_):
		return self.has_atleast(min(self.cluster_size,self.max_cluster_size), type_)

	def is_valid_cluster(self, type_id):
		if type_id not in self.types:
			return False
		return self.has_atleast(self.min_cluster_size, self.get_type(type_id))

	def remove_less_important_batches(self, n):
		# Pick the right tree list
		if random() <= self._prioritized_drop_probability: 
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
			for type_ in self.type_values
			if tree_list[type_].inserted_elements >= self.min_cluster_size
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
		
	def add(self, batch, type_id=0, on_policy=False): # O(log)
		self._add_type_if_not_exist(type_id)
		type_ = self.get_type(type_id)
		type_batch = self.batches[type_]
		idx = None
		if self.is_full_cluster(type_): # full cluster, remove from it
			idx = self.get_less_important_batch(type_)
		elif self.is_full_buffer(): # full buffer but not full cluster, remove the less important batch in the whole buffer
			self.remove_less_important_batches(1)
		if idx is None: # add new element to buffer
			idx = len(type_batch)
			type_batch.append(batch)
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
			self._insertion_time_tree[type_][idx] = (time.time(), idx) # O(log)
		# Set drop priority
		if self._prioritized_drop_probability > 0 and self._global_distribution_matching:
			self._drop_priority_tree[type_][idx] = (random(), idx) # O(log)
		# Set priority
		self.update_priority(batch, idx, type_id) # add batch
		# if self._beta and 'weights' not in batch: # Add default weights
		# 	batch['weights'] = np.ones(batch.count, dtype=np.float32)
		if self._beta and on_policy: # Update weights after updating priority
			self.update_beta_weights(batch, idx, type_)
		if self.global_size:
			assert self.count() <= self.global_size, 'Memory leak in replay buffer; v1'
			assert super().count() <= self.global_size, 'Memory leak in replay buffer; v2'
		return idx, type_id

	def sample_cluster(self):
		tree_list = [
			(i,t) 
			for i,t in enumerate(self._sample_priority_tree) 
			if t.inserted_elements > 0
		]
		if self._cluster_prioritisation_strategy is not None and len(tree_list)>1:
			type_priority = map(lambda x: self.get_cluster_priority(x[-1]), tree_list)
			# type_priority = map(self.normalize_priority, type_priority)
			type_priority = np.array(tuple(type_priority))
			if self._cluster_prioritisation_strategy == 'average':
				avg_type_priority = np.mean(type_priority)
				type_priority = -np.absolute(type_priority-avg_type_priority) # the closer to the average, the higher the priority: the hardest tasks will be tackled last
			elif self._cluster_prioritisation_strategy == 'above_average':
				avg_type_priority = np.mean(type_priority)
				type_priority_above_avg = type_priority[type_priority>avg_type_priority]
				best_after_mean = np.min(type_priority_above_avg) if type_priority_above_avg.size > 0 else type_priority[0]
				type_priority = -np.absolute(type_priority-best_after_mean) # the closer to the best_after_mean, the higher the priority: the hardest tasks will be tackled last
			def smooth_eta_normalisation(x): 
				x_min = np.min(x)
				x_max = np.max(x)
				x_eta = np.abs(np.std(x)/x_max)*self._cluster_prioritisation_smoothing_degree # eta is proportional to the standard deviation of x, otherwise the elements with the lowest priority would almost never be sampled
				return self.eta_normalisation(x, x_min, x_max, x_eta)
			type_priority = smooth_eta_normalisation(smooth_eta_normalisation(type_priority)) # first eta-normalisation makes priorities in (0,1], but it inverts their magnitude # second eta-normalisation guarantees that original priorities' magnitude is preserved
			# print(tuple(zip(list(type_priority),self.type_keys)))
			type_cumsum = np.cumsum(type_priority) # O(|self.type_keys|)
			type_mass = random() * type_cumsum[-1] # O(1)
			assert 0 <= type_mass, f'type_mass {type_mass} should be greater than 0'
			assert type_mass <= type_cumsum[-1], f'type_mass {type_mass} should be lower than {type_cumsum[-1]}'
			tree_idx,_ = next(filter(lambda x: x[-1] >= type_mass, enumerate(type_cumsum))) # O(|self.type_keys|)
			type_ = tree_list[tree_idx][0]
		else:
			type_ = choice(tree_list)[0]
		type_id = self.type_keys[type_]
		return type_id, type_

	def sample(self, n=1): # O(log)
		type_id, type_ = self.sample_cluster()
		type_sum_tree = self._sample_priority_tree[type_]
		type_batch = self.batches[type_]
		idx_list = [
			type_sum_tree.find_prefixsum_idx(prefixsum_fn=lambda mass: mass*random(), check_min=self._priority_can_be_negative) # O(log)
			for _ in range(n)
		]
		batch_list = [
			type_batch[idx] # O(1)
			for idx in idx_list
		]
		# Update weights
		if self._beta: # Update weights
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
		return (upper_max_priority - priorities)/(upper_max_priority - min_priority) # in (0,1]: the closer is type_sum_tree[idx] to max_priority, the lower is the weight

	def update_beta_weights(self, batch, idx, type_):
		type_sum_tree = self._sample_priority_tree[type_]
		if self._cluster_level_weighting: min_priority = type_sum_tree.min_tree.min()[0] # O(log)
		else: min_priority = np.mean(tuple(map(lambda x: x.min_tree.min()[0], self._sample_priority_tree))) # O(log) # Using the average min priority we are trying to smooth the effect of outliers, that are hard to be removed from the buffer.
		batch_priority = type_sum_tree[idx]
		batch_priority = max(min_priority, batch_priority)
		if self._priority_can_be_negative:
			if self._cluster_level_weighting: max_priority = type_sum_tree.max_tree.max()[0] # O(log)
			else: max_priority = max(map(lambda x: x.max_tree.max()[0], self._sample_priority_tree)) # O(log)
			weight = self.eta_normalisation(batch_priority, min_priority, max_priority, self._eta)
			# print(weight, max_priority-min_priority)
		else:
			assert min_priority > 0, f"min_priority must be > 0, if beta is not None and priority_can_be_negative is False, but it is {min_priority}"
			weight = min_priority / batch_priority # default, not compatible with negative priorities # in (0,1]: the closer is type_sum_tree[idx] to max_priority, the lower is the weight
		weight = weight**self._beta
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

	def stats(self, debug=False):
		stats_dict = super().stats(debug)
		stats_dict.update({
			'cluster_capacity':self.get_cluster_capacity_dict(),
			'cluster_priority': self.get_cluster_priority_dict(),
		})
		return stats_dict
