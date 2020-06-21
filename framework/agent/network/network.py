# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
from more_itertools import unique_everseen

class Network():
	def __init__(self, id, training):
		self.training = training
		self.id = id
		self.use_internal_state = False
		# Initialize keys collections
		self.shared_keys = []
		self.update_keys = []
	
	def _update_keys(self, scope_name, share_trainables):
		if share_trainables:
			self.shared_keys = list(unique_everseen(self.shared_keys + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)))
		self.update_keys = list(unique_everseen(self.update_keys + tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope_name)))

	def _scopefy(self, output_fn, layer_type, scope, name, share_trainables):
		with tf.variable_scope("{}/{}-{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			output = output_fn()
			self._update_keys(variable_scope.name, share_trainables)
			# print( "	[{}]{} layer shape: {}".format(self.id, layer_type, output.get_shape()) )
			return output
		
	def _batch_normalization_layer(self, input, scope, name="", share_trainables=True, renorm=False, center=True, scale=True):
		def layer_fn():
			batch_norm = tf.layers.BatchNormalization(renorm=renorm, center=center, scale=scale) # renorm when minibaches are too small
			norm_input = batch_norm.apply(input, training=self.training)
			return batch_norm, norm_input
		return self._scopefy(output_fn=layer_fn, layer_type='BatchNorm', scope=scope, name=name, share_trainables=share_trainables)
			
	def _feature_entropy_layer(self, input, scope, name="", share_trainables=True): # feature entropy measures how much the input is uncommon
		batch_norm, _ = self._batch_normalization_layer(input=input, scope=scope, name=layer_type)
		def layer_fn():
			fentropy = Normal(batch_norm.moving_mean, tf.sqrt(batch_norm.moving_variance)).cross_entropy(input)
			fentropy = tf.layers.flatten(fentropy)
			if len(fentropy.get_shape()) > 1:
				fentropy = tf.reduce_mean(fentropy, axis=-1)
			return fentropy
		return self._scopefy(output_fn=layer_fn, layer_type='Fentropy', scope=scope, name=name, share_trainables=share_trainables)			
