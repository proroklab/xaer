# -*- coding: utf-8 -*-
import options
flags = options.get()

import numpy as np
import tensorflow.compat.v1 as tf
from agent.network.base_network import Base_Network
import utils.tensorflow_utils as tf_utils

class Impala_Network(Base_Network):
	"""
	Model used in the paper "IMPALA: Scalable Distributed Deep-RL with 
	Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
	"""
	dropout_probability = 0.
	use_batch_norm = False
	# depths = [32, 64, 64, 64, 64] # Large
	depths=[16, 32, 32] # Small

	def conv_layer(self, out, depth):
		out = tf.layers.conv2d(out, depth, 3, padding='same')
		if self.dropout_probability > 0:
			out = tf.layers.dropout(inputs=out, rate=self.dropout_probability)
		if self.use_batch_norm:
			out = tf.contrib.layers.batch_norm(out, center=True, scale=True, is_training=True)
		return out
		
	def residual_block(self, inputs):
		depth = inputs.get_shape()[-1].value
		out = tf.nn.relu(inputs)
		out = self.conv_layer(out, depth)
		out = tf.nn.relu(out)
		out = self.conv_layer(out, depth)
		return out + inputs

	def conv_sequence(self, inputs, depth):
		out = self.conv_layer(inputs, depth)
		out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
		out = self.residual_block(out)
		out = self.residual_block(out)
		return out

	def _cnn_layer(self, input, scope, name="", share_trainables=True):
		layer_type = 'CNN'
		def layer_fn():
			xx = input
			for depth in self.depths:
				xx = self.conv_sequence(xx, depth)
			xx = tf.keras.layers.Flatten()(xx)
			xx = tf.nn.relu(xx)
			return xx
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def _concat_layer(self, input, concat, scope, name="", share_trainables=True):
		layer_type = 'Concat'
		def layer_fn():
			xx = tf.keras.layers.Flatten()(input)
			if concat.get_shape()[-1] > 0:
				xx = tf.concat([xx, tf.keras.layers.Flatten()(concat)], -1) # shape: (batch, concat_size+units)
			xx = tf.keras.layers.Dense(name='Concat_Dense1',  units=256, activation=tf.nn.relu)(xx)
			return xx
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)
