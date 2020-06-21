# -*- coding: utf-8 -*-
import options
flags = options.get()

import numpy as np
import tensorflow.compat.v1 as tf
from agent.network.openai_small_network import OpenAISmall_Network
import utils.tensorflow_utils as tf_utils
from utils.rnn import RNN

# N.B. tf.initializers.orthogonal is broken with tensorflow 1.10 and GPU, use OpenAI implementation

class OpenAISmallCRELU_Network(OpenAISmall_Network):

	def _cnn_layer(self, input, scope, name="", share_trainables=True):
		layer_type = 'CNN'
		def layer_fn():
			# input = tf.contrib.model_pruning.masked_conv2d(inputs=input, num_outputs=16, kernel_size=(3,3), padding='SAME', activation_fn=tf.nn.leaky_relu) # xavier initializer
			# input = tf.contrib.model_pruning.masked_conv2d(inputs=input, num_outputs=32, kernel_size=(3,3), padding='SAME', activation_fn=tf.nn.leaky_relu) # xavier initializer
			xx = tf.keras.layers.Conv2D(name='CNN_Conv1',  filters=32, kernel_size=8, strides=4, padding='SAME', activation=tf.nn.crelu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))(input)
			xx = tf.keras.layers.Conv2D(name='CNN_Conv2',  filters=64, kernel_size=4, strides=2, padding='SAME', activation=tf.nn.crelu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))(xx)
			xx = tf.keras.layers.Conv2D(name='CNN_Conv3',  filters=64, kernel_size=4, strides=1, padding='SAME', activation=tf.nn.crelu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))(xx)
			return xx
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def _weights_layer(self, input, weights, scope, name="", share_trainables=True):
		layer_type = 'Weights'
		def layer_fn():
			kernel = tf.stop_gradient(weights['kernel'])
			kernel = tf.transpose(kernel, [1, 0])
			kernel = tf.keras.layers.Dense(name='TS_Dense0',  units=1, activation=tf.nn.crelu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))(kernel)
			kernel = tf.reshape(kernel, [1,-1])
			bias = tf.stop_gradient(weights['bias'])
			bias = tf.reshape(bias, [1,-1])
			weight_state = tf.concat((kernel, bias), -1)
			weight_state = tf.layers.dense(name='TS_Dense1', inputs=weight_state, units=64, activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))
			weight_state = tf.reshape(weight_state, [-1])
			xx = tf.layers.flatten(input)
			xx = tf.map_fn(fn=lambda b: tf.concat((b,weight_state),-1), elems=xx)
			return xx
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)
	