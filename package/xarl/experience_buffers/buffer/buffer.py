# -*- coding: utf-8 -*-
import logging
from random import choice
from collections import deque
import uuid

logger = logging.getLogger(__name__)

class Buffer(object):
	__slots__ = ('cluster_size','global_size','types','batches','type_values','type_keys')
	
	def __init__(self, cluster_size=None, global_size=50000, **args):
		assert cluster_size or global_size, 'At least one of cluster_size or global_size shall be set greater than 0.'
		if not cluster_size: cluster_size = global_size
		self.cluster_size = min(cluster_size,global_size) if global_size else cluster_size
		self.global_size = global_size
		self.clean()
		
	def clean(self):
		self.types = {}
		self.type_values = []
		self.type_keys = [] 
		self.batches = []
		
	def _add_type_if_not_exist(self, type_id): # private method
		if type_id in self.types: # check it to avoid double insertion
			return False
		self.types[type_id] = sample_type = len(self.type_keys)
		self.type_values.append(sample_type)
		self.type_keys.append(type_id)
		self.batches.append(deque(maxlen=self.cluster_size))
		logger.warning(f'Added a new cluster with id {type_id}, now there are {len(self.type_values)} different clusters.')
		return True
		
	def set(self, buffer):
		assert isinstance(buffer, Buffer)
		for key in self.__slots__:
			setattr(self, key, getattr(buffer, key))
		
	def get_batches(self, type_id=None):
		if type_id is None:
			result = []
			for batch in self.batches:
				result += batch
			return result
		return self.batches[self.get_type(type_id)]

	def has_atleast(self, frames, type_=None):
		return self.count(type_) >= frames
		
	def has(self, frames, type_=None):
		return self.count(type_) == frames
		
	def count(self, type_=None):
		if type_ is None:
			if len(self.batches) == 0:
				return 0
			return sum(len(batch) for batch in self.batches)
		return len(self.batches[type_])
		
	def id_is_full(self, type_id):
		return self.has(self.cluster_size, self.get_type(type_id))
		
	def is_full_cluster(self, type_=None):
		if type_ is None:
			return self.has(self.cluster_size*len(self.types))
		return self.has(self.cluster_size, type_)

	def is_full_buffer(self):
		return self.has(self.global_size) if self.global_size else False

	def is_full(self, type_=None):
		return self.is_full_cluster(type_) or self.is_full_buffer()
		
	def is_empty(self, type_=None):
		return not self.has_atleast(1, type_)
		
	def get_type(self, type_id):
		return self.types[type_id]

	def add(self, batch, type_id=0, **args): # put batch into buffer
		self._add_type_if_not_exist(type_id)
		type_ = self.get_type(type_id)
		batch["infos"][0]["batch_uid"] = str(uuid.uuid4()) # random unique id
		if self.is_full_buffer():
			biggest_cluster = max(self.type_values, key=self.count)
			self.batches[biggest_cluster].popleft()
		self.batches[type_].append(batch)

	def sample(self, n=1):
		type_ = choice(self.type_values)
		batch_list = [
			choice(self.batches[type_])
			for _ in range(n)
		]
		return batch_list

	def stats(self, debug=False):
		return {
			"added_count": self.count(),
		}
