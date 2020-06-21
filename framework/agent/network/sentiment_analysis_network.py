# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
from agent.network.base_network import Base_Network

class SA_Network(Base_Network):
	lstm_units = 128 # the number of units of the LSTM
	
	def _cnn_layer(self, input, scope, name="", share_trainables=True):
		layer_type = 'CNN'
		def layer_fn():
			xx = tf.keras.layers.Conv2D(name='CNN_Conv1',  filters=16, kernel_size=(1,3), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling )(input)
			return xx
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def _concat_layer(self, input, concat, scope, name="", share_trainables=True):
		layer_type = 'Concat'
		def layer_fn():
			xx = tf.layers.flatten(input)
			xx = tf.keras.layers.Dense(name='Concat_Dense1',  units=128, activation=None, kernel_initializer=tf.initializers.variance_scaling)(xx)
			xx = tf.contrib.layers.maxout(inputs=xx, num_units=64, axis=-1)
			xx = tf.reshape(xx, [-1, 64])
			if concat.get_shape()[-1] > 0:
				concat = tf.layers.flatten(concat)
				xx = tf.concat([xx, concat], -1) # shape: (batch, concat_size+units)
			return xx
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)
