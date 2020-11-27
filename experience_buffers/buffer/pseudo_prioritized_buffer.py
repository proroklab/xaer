# -*- coding: utf-8 -*-
from random import choice, random, randint
import numpy as np
import time
from experience_buffers.buffer.buffer import Buffer
from utils.segment_tree import SumSegmentTree, MinSegmentTree

class PseudoPrioritizedBuffer(Buffer):
	__slots__ = ('_alpha','_prioritized_drop_probability','_epsilon','_global_distribution_matching','_it_capacity','_sample_priority_tree','_drop_priority_tree','_insertion_time_tree','_prioritised_cluster_sampling')
	
	def __init__(self, size, alpha=1, prioritized_drop_probability=0.5, global_distribution_matching=False, prioritised_cluster_sampling=True): # O(1)
		self._epsilon = 1e-6
		self._alpha = alpha # how much prioritization is used (0 - no prioritization, 1 - full prioritization)
		self._it_capacity = 1
		self._prioritized_drop_probability = prioritized_drop_probability # remove the worst batch with this probability otherwise remove the oldest one
		self._global_distribution_matching = global_distribution_matching
		self._prioritised_cluster_sampling = prioritised_cluster_sampling
		while self._it_capacity < size:
			self._it_capacity *= 2
		super().__init__(size)
		
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
		self.types[type_id] = len(self.types)
		self.type_values.append(self.types[type_id])
		self.type_keys.append(type_id)
		self.batches.append([])
		self._sample_priority_tree.append(SumSegmentTree(self._it_capacity))
		self._drop_priority_tree.append(MinSegmentTree(self._it_capacity,neutral_element=(float('inf'),-1)))
		self._insertion_time_tree.append(MinSegmentTree(self._it_capacity,neutral_element=(float('inf'),-1)))
		return True
	
	def normalize_priority(self, priority): # O(1)
		priority_sign = -1 if priority < 0 else 1
		priority = np.absolute(priority) + self._epsilon 
		return priority_sign*np.power(priority, self._alpha).astype('float32')

	def get_priority(self, idx, type_id):
		sample_type = self.get_type(type_id)
		return self._sample_priority_tree[sample_type][idx]
		
	def add(self, batch, priority, type_id=0): # O(log)
		self._add_type_if_not_exist(type_id)
		sample_type = self.get_type(type_id)
		type_batch = self.batches[sample_type]
		if self.is_full(sample_type): # full buffer
			if random() <= self._prioritized_drop_probability: # Remove the batch with lowest priority
				_,idx = self._drop_priority_tree[sample_type].min() # O(1)
			else: # Remove the oldest batch
				_,idx = self._insertion_time_tree[sample_type].min() # O(1)
			type_batch[idx] = batch
		else: # add new element to buffer
			idx = len(type_batch)
			type_batch.append(batch)
		batch["batch_indexes"] = np.array([idx]*batch.count)
		# batch["batch_types"] = np.array([type_id]*batch.count)
		# Set insertion time
		if self._prioritized_drop_probability < 1:
			self._insertion_time_tree[sample_type][idx] = (time.time(), idx) # O(log)
		# Set drop priority
		if self._prioritized_drop_probability > 0 and self._global_distribution_matching:
			self._drop_priority_tree[sample_type][idx] = (random(), idx) # O(log)
		# Set priority
		self.update_priority(idx, priority, type_id)
		return idx, type_id

	def sample_cluster(self):
		if self._prioritised_cluster_sampling:
			type_priority = list(map(lambda x: x.sum(scaled=False), self._sample_priority_tree))
			worse_type_priority = min(type_priority)
			type_cumsum = np.cumsum(list(map(lambda x: x-worse_type_priority, type_priority))) # O(|self.type_keys|)
			type_mass = random() * type_cumsum[-1] # O(1)
			sample_type,_ = next(filter(lambda x: x[-1] >= type_mass, enumerate(type_cumsum))) # O(|self.type_keys|)
			type_id = self.type_keys[sample_type]
		else:
			type_id = choice(self.type_keys)
			sample_type = self.get_type(type_id)
		return type_id, sample_type
		
	def sample(self, remove=False): # O(log)
		type_id, sample_type = self.sample_cluster()
		# print(type_id)
		type_sum_tree = self._sample_priority_tree[sample_type]
		idx = type_sum_tree.find_prefixsum_idx(prefixsum_fn=lambda mass: mass*random()) # O(log)
		type_batch = self.batches[sample_type]
		idx = np.clip(idx, 0,len(type_batch)-1)
		# Remove from buffer
		if remove:
			self._insertion_time_tree[sample_type][idx] = None # O(log)
			self._drop_priority_tree[sample_type][idx] = None # O(log)
			self._sample_priority_tree[sample_type][idx] = None # O(log)
		return type_batch[idx]
	
	def update_priority(self, idx, priority, type_id=0): # O(log)
		sample_type = self.get_type(type_id)
		normalized_priority = self.normalize_priority(priority)
		# Update priority
		if self._prioritized_drop_probability > 0 and not self._global_distribution_matching:
			self._drop_priority_tree[sample_type][idx] = (normalized_priority, idx) # O(log)
		self._sample_priority_tree[sample_type][idx] = normalized_priority # O(log)
