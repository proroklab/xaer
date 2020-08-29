# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf

from more_itertools import unique_everseen
import os

class Network():
	scope_cache = {}

	def __init__(self, id, training):
		self.training = training
		self.id = id
		self.use_internal_state = False
		# Initialize keys collections
		self.shared_keys = []
		self.update_keys = []

	@staticmethod
	def format_scope_name(parts):
		return os.path.join(*map(lambda x: str(x).strip('/'), parts)).strip('/')
	
	def _update_keys(self, scope_name, share_trainables):
		if share_trainables:
			self.shared_keys = list(unique_everseen(self.shared_keys + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)))
		self.update_keys = list(unique_everseen(self.update_keys + tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope_name)))

	def _scopefy(self, inputs, output_fn, layer_type, scope, name, share_trainables, reuse=True):
		with tf.variable_scope(self.format_scope_name([scope,layer_type,name])) as variable_scope:
			scope_name = variable_scope.name
			if reuse:
				reusing = True
				layer = self.scope_cache.get(scope_name, None)
				if layer is None:
					reusing = False
					layer = self.scope_cache[scope_name] = output_fn()
			else:
				reusing = False
				print( "	[{}]Building scope: {}".format(self.id, scope_name) )
				layer = output_fn()
			output = layer(*inputs)
			if not reusing:
				self._update_keys(scope_name, share_trainables)
				print( "	[{}]Building scope: {}".format(self.id, scope_name) )
			else:
				print( "	[{}]Reusing scope: {}".format(self.id, scope_name) )
			# print( "	[{}]{} layer shape: {}".format(self.id, layer_type, output.get_shape()) )
			return output
