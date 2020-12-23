# -*- coding: utf-8 -*-
from random import choice, random, randint
import numpy as np
import time
from xarl.experience_buffers.buffer.buffer import Buffer
from xarl.utils.segment_tree import SumSegmentTree, MinSegmentTree
import copy

class PseudoPrioritizedBuffer(Buffer):
	__slots__ = ('_priority_id','_priority_aggregation_fn','_alpha','_beta','_epsilon','_prioritized_drop_probability','_global_distribution_matching','_it_capacity','_sample_priority_tree','_drop_priority_tree','_insertion_time_tree','_prioritised_cluster_sampling','_sample_simplest_unknown_task')
	
	def __init__(self, 
		priority_id,
		priority_aggregation_fn,
		cluster_size=50000, 
		global_size=50000, 
		alpha=0.6, 
		beta=0.4, 
		epsilon=1e-4,
		prioritized_drop_probability=0.5, 
		global_distribution_matching=False, 
		prioritised_cluster_sampling=True, 
		sample_simplest_unknown_task=False,
	): # O(1)
		self._priority_id = priority_id
		self._priority_aggregation_fn = eval(priority_aggregation_fn)
		self._alpha = alpha # How much prioritization is used (0 - no prioritization, 1 - full prioritization)
		self._beta = beta # To what degree to use importance weights (0 - no corrections, 1 - full correction).
		assert self._beta is None or self._beta >= 0., "beta >= 0, if beta is not None"
		self._epsilon = epsilon # Epsilon to add to the priorities when updating priorities.
		self._prioritized_drop_probability = prioritized_drop_probability # remove the worst batch with this probability otherwise remove the oldest one
		self._global_distribution_matching = global_distribution_matching
		self._prioritised_cluster_sampling = prioritised_cluster_sampling
		self._sample_simplest_unknown_task = sample_simplest_unknown_task
		self._it_capacity = 1
		while self._it_capacity < cluster_size:
			self._it_capacity *= 2
		super().__init__(cluster_size, global_size)
		
	def set(self, buffer): # O(1)
		assert isinstance(buffer, PseudoPrioritizedBuffer)
		super().set(buffer)
	
	def clean(self): # O(1)
		super().clean()
		self._sample_priority_tree = []
		self._drop_priority_tree = []
		self._insertion_time_tree = []
			
	def _add_type_if_not_exist(self, type_id): # O(1)
		if type_id in self.types: # check it to avoid double insertion
			return False
		self.types[type_id] = sample_type = len(self.types)
		self.type_values.append(sample_type)
		self.type_keys.append(type_id)
		self.batches.append([])
		self._sample_priority_tree.append(SumSegmentTree(self._it_capacity))
		self._drop_priority_tree.append(MinSegmentTree(self._it_capacity,neutral_element=(float('inf'),-1)))
		self._insertion_time_tree.append(MinSegmentTree(self._it_capacity,neutral_element=(float('inf'),-1)))
		return True
	
	def normalize_priority(self, priority): # O(1)
		# always add self._epsilon so that there is no priority equal to the neutral value of a SumSegmentTree
		return np.sign(priority)*(np.absolute(priority) + self._epsilon)**self._alpha

	def get_priority(self, idx, type_id):
		sample_type = self.get_type(type_id)
		return self._sample_priority_tree[sample_type][idx]

	def remove_batch(self, sample_type, idx): # O(log)
		last_idx = len(self.batches[sample_type])-1
		if idx == last_idx:
			if self._prioritized_drop_probability > 0:
				self._drop_priority_tree[sample_type][idx] = None # O(log)
			if self._prioritized_drop_probability < 1:
				self._insertion_time_tree[sample_type][idx] = None # O(log)
			self._sample_priority_tree[sample_type][idx] = None # O(log)
			self.batches[sample_type].pop()
		elif idx < last_idx:
			if self._prioritized_drop_probability > 0:
				self._drop_priority_tree[sample_type][idx] = (self._drop_priority_tree[sample_type][last_idx][0],idx) # O(log)
				self._drop_priority_tree[sample_type][last_idx] = None # O(log)
			if self._prioritized_drop_probability < 1:
				self._insertion_time_tree[sample_type][idx] = (self._insertion_time_tree[sample_type][last_idx][0],idx) # O(log)
				self._insertion_time_tree[sample_type][last_idx] = None # O(log)
			self._sample_priority_tree[sample_type][idx] = self._sample_priority_tree[sample_type][last_idx] # O(log)
			self._sample_priority_tree[sample_type][last_idx] = None # O(log)
			self.batches[sample_type][idx] = self.batches[sample_type].pop()

	def count(self, type_=None):
		if type_ is None:
			if len(self.batches) == 0:
				return 0
			return sum(t.inserted_elements for t in self._sample_priority_tree)
		return self._sample_priority_tree[type_].inserted_elements

	def get_less_important_batch(self, sample_type):
		if random() <= self._prioritized_drop_probability: # Remove the batch with lowest priority
			_,idx = self._drop_priority_tree[sample_type].min() # O(1)
		else:
			_,idx = self._insertion_time_tree[sample_type].min() # O(1)
		return idx

	def remove_less_important_batches(self, sample_type_list):
		for sample_type in sample_type_list:
			# if not self.has_atleast(2, sample_type): # avoid empty clusters
			# 	continue
			idx = self.get_less_important_batch(sample_type)
			self.remove_batch(sample_type, idx)
		
	def add(self, batch, type_id=0): # O(log)
		self._add_type_if_not_exist(type_id)
		sample_type = self.get_type(type_id)
		type_batch = self.batches[sample_type]
		idx = None
		if self.is_full_buffer(): # full buffer, remove from the biggest cluster
			biggest_cluster = max(self.type_values, key=self.count)
			if sample_type == biggest_cluster:
				idx = self.get_less_important_batch(sample_type)
			else:
				self.remove_less_important_batches([biggest_cluster])
		elif self.is_full_cluster(sample_type): # full buffer
			# self.remove_less_important_batches([sample_type])
			idx = self.get_less_important_batch(sample_type)
		if idx is None: # add new element to buffer
			idx = len(type_batch)
			type_batch.append(batch)
		batch['infos'][0]["batch_index"][type_id] = idx
		if self._beta is not None and 'weights' not in batch: # Add default weights
			batch['weights'] = np.ones(batch.count, dtype=np.float32)
		# batch["batch_types"] = np.array([type_id]*batch.count)
		# Set insertion time
		if self._prioritized_drop_probability < 1:
			self._insertion_time_tree[sample_type][idx] = (time.time(), idx) # O(log)
		# Set drop priority
		if self._prioritized_drop_probability > 0 and self._global_distribution_matching:
			self._drop_priority_tree[sample_type][idx] = (random(), idx) # O(log)
		# Set priority
		self.update_priority(batch, idx, type_id) # add batch
		# assert not self.global_size or self.count() <= self.global_size, 'Memory leak in replay buffer; v1'
		# assert not self.global_size or sum(len(b) for b in self.batches) <= self.global_size, 'Memory leak in replay buffer; v2'
		return idx, type_id

	def sample_cluster(self):
		if self._prioritised_cluster_sampling:
			type_priority = np.array(tuple(map(lambda x: x.sum(scaled=False), self._sample_priority_tree)), dtype=np.float32)
			if self._sample_simplest_unknown_task == 'average':
				avg_type_priority = np.mean(type_priority)
				type_priority = -np.absolute(type_priority-avg_type_priority) # the closer to the average, the higher the priority: the hardest tasks will be tackled last
			elif self._sample_simplest_unknown_task == 'above_average':
				avg_type_priority = np.mean(type_priority)
				type_priority_above_avg = type_priority[type_priority>avg_type_priority]
				best_after_mean = np.min(type_priority_above_avg) if type_priority_above_avg.size > 0 else type_priority[0]
				type_priority = -np.absolute(type_priority-best_after_mean) # the closer to the best_after_mean, the higher the priority: the hardest tasks will be tackled last
			worst_type_priority = np.min(type_priority)
			type_cumsum = np.cumsum(type_priority-worst_type_priority) # O(|self.type_keys|)
			type_mass = random() * type_cumsum[-1] # O(1)
			sample_type,_ = next(filter(lambda x: x[-1] >= type_mass, enumerate(type_cumsum))) # O(|self.type_keys|)
			type_id = self.type_keys[sample_type]
		else:
			type_id = choice(self.type_keys)
			sample_type = self.get_type(type_id)
		return type_id, sample_type

	def sample(self, n=1, remove=False): # O(log)
		type_id, sample_type = self.sample_cluster()
		type_sum_tree = self._sample_priority_tree[sample_type]
		type_batch = self.batches[sample_type]
		if self._beta is not None: # Update weights
			min_priority = type_sum_tree.min_tree.min()
			assert min_priority > 0, "min_priority > 0, if beta is not None"
		batch_list = []
		for _ in range(n):
			idx = type_sum_tree.find_prefixsum_idx(prefixsum_fn=lambda mass: mass*random()) # O(log)
			#idx = np.clip(idx, 0,len(type_batch)-1)
			batch = type_batch[idx]
			# Update weights
			if self._beta is not None: # Update weights
				weight = (min_priority / type_sum_tree[idx])**self._beta
				batch['weights'] = np.full(batch.count, weight, dtype=np.float32)
			# Remove from buffer
			if remove is True:
				self.remove_batch(sample_type, idx)
			batch_list.append(batch)
		return batch_list

	def get_batch_priority(self, batch):
		return self._priority_aggregation_fn(batch[self._priority_id])
	
	def update_priority(self, new_batch, idx, type_id=0): # O(log)
		# new_batch = new_batch.copy()
		# new_batch.compress()
		# for k,v in new_batch.data.items():
		# 	new_batch[k] = copy.deepcopy(v)
		sample_type = self.get_type(type_id)
		if idx >= len(self.batches[sample_type]):
			return
		self.batches[sample_type][idx] = new_batch
		new_priority = self.get_batch_priority(new_batch)
		normalized_priority = self.normalize_priority(new_priority)
		# Update priority
		if self._prioritized_drop_probability > 0 and not self._global_distribution_matching:
			self._drop_priority_tree[sample_type][idx] = (normalized_priority, idx) # O(log)
		self._sample_priority_tree[sample_type][idx] = normalized_priority # O(log)
