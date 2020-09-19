# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
from agent.network.network import Network
from tf_agents.utils import common

class Singular_Network(Network):
	def __init__(self, id, name, scope_dict, training=True):
		super().__init__(id, training)
		self.name = name
		self.scope_name = scope_dict['self']
		
	def get(self, initial_value, scope=None, name=None, share_trainables=True):
		layer_type = 'Singular'
		def layer_fn():
			var = common.create_variable('singular_var', initial_value=initial_value, dtype=tf.float32, trainable=True)
			def exec_fn(i):
				return var
			return exec_fn
		return self._scopefy(inputs=(initial_value,), output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)
