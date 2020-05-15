# -*- coding: utf-8 -*-
from random import choice, random, randint
import numpy as np
import time
from utils.buffer.buffer import Buffer
from utils.segment_tree import SumSegmentTree, MinSegmentTree

class PseudoPrioritizedBuffer(Buffer):
	__slots__ = ('_alpha','_prioritized_drop_probability','_epsilon','_it_capacity','_it_sum','_it_min','_it_time')
	
	def __init__(self, size, alpha=1, prioritized_drop_probability=0.5): # O(1)
		self._epsilon = 1e-6
		self._alpha = alpha # how much prioritization is used (0 - no prioritization, 1 - full prioritization)
		self._it_capacity = 1
		self._prioritized_drop_probability = prioritized_drop_probability # remove the worst batch with this probability otherwise remove the oldest one
		while self._it_capacity < size:
			self._it_capacity *= 2
		super().__init__(size)
		
	def set(self, buffer): # O(1)
		assert isinstance(buffer, PseudoPrioritizedBuffer)
		super().set(buffer)
	
	def clean(self): # O(1)
		super().clean()
		self._it_sum = []
		self._it_min = []
		self._it_time = []
			
	def _add_type_if_not_exist(self, type_id): # O(1)
		if type_id in self.types: # check it to avoid double insertion
			return False
		self.types[type_id] = len(self.types)
		self.type_values.append(self.types[type_id])
		self.type_keys.append(type_id)
		self.batches.append([])
		self._it_sum.append(SumSegmentTree(self._it_capacity))
		self._it_min.append(MinSegmentTree(self._it_capacity,neutral_element=(float('inf'),-1)))
		self._it_time.append(MinSegmentTree(self._it_capacity,neutral_element=(float('inf'),-1)))
		return True
	
	def normalize_priority(self, priority): # O(1)
		priority_sign = -1 if priority < 0 else 1
		priority = np.absolute(priority) + self._epsilon 
		return priority_sign*np.power(priority, self._alpha)
		
	def put(self, batch, priority, type_id=0): # O(log)
		self._add_type_if_not_exist(type_id)
		sample_type = self.get_type(type_id)
		type_batch = self.batches[sample_type]
		if self.is_full(sample_type): # full buffer
			if random() <= self._prioritized_drop_probability: # Remove the batch with lowest priority
				_,idx = self._it_min[sample_type].min() # O(1)
			else: # Remove the oldest batch
				_,idx = self._it_time[sample_type].min() # O(1)
			type_batch[idx] = batch
		else: # add new element to buffer
			idx = len(type_batch)
			type_batch.append(batch)
		# Update time
		self._it_time[sample_type][idx] = (time.time(), idx) # O(log)
		# Update priority
		self.update_priority(idx, priority, type_id)
		
	def keyed_sample(self): # O(log)
		type_id = choice(self.type_keys)
		sample_type = self.get_type(type_id)
		type_sum_tree = self._it_sum[sample_type]
		mass = random() * type_sum_tree.sum() # O(1)
		idx = type_sum_tree.find_prefixsum_idx(mass) # O(log)
		type_batch = self.batches[sample_type]
		idx = np.clip(idx, 0,len(type_batch)-1)
		# weight = (self._it_sum[idx]/self._it_min.min()) ** (-beta) # importance weight
		# return self.batches[0][idx], idx, weight # multiply weight for advantage
		return type_batch[idx], idx, type_id
	
	def sample(self): # O(log)
		return self.keyed_sample()[0]
	
	def update_priority(self, idx, priority, type_id=0): # O(log)
		sample_type = self.get_type(type_id)
		normalized_priority = self.normalize_priority(priority)
		# Update min
		self._it_min[sample_type][idx] = (normalized_priority, idx) # O(log)
		# Update priority
		self._it_sum[sample_type][idx] = normalized_priority # O(log)
