# -*- coding: utf-8 -*-
import options
flags = options.get()

import numpy as np
import tensorflow.compat.v1 as tf

from agent.network.openai_small_network import OpenAISmall_Network
import utils.tensorflow_utils as tf_utils
# from utils.rnn import RNN

# N.B. tf.initializers.orthogonal is broken with tensorflow 1.10 and GPU, use OpenAI implementation

class OpenAISmallCRELU_Network(OpenAISmall_Network):

	def _cnn_layer(self, input, scope=None, name=None, share_trainables=True):
		layer_type = 'CNN'
		def layer_fn():
			input_layer = tf.keras.Sequential(name=layer_type, layers=[
				# input = tf.contrib.model_pruning.masked_conv2d(inputs=input, num_outputs=16, kernel_size=(3,3), padding='SAME', activation_fn=tf.nn.leaky_relu) # xavier initializer
				# input = tf.contrib.model_pruning.masked_conv2d(inputs=input, num_outputs=32, kernel_size=(3,3), padding='SAME', activation_fn=tf.nn.leaky_relu) # xavier initializer
				tf.keras.layers.Conv2D(name='CNN_Conv1',  filters=32, kernel_size=8, strides=4, padding='SAME', activation=tf.nn.crelu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2))),
				tf.keras.layers.Conv2D(name='CNN_Conv2',  filters=64, kernel_size=4, strides=2, padding='SAME', activation=tf.nn.crelu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2))),
				tf.keras.layers.Conv2D(name='CNN_Conv3',  filters=64, kernel_size=4, strides=1, padding='SAME', activation=tf.nn.crelu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2))),
			])
			def exec_fn(i):
				return input_layer(i)
			return exec_fn
		return self._scopefy(inputs=(input, ), output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def _weights_layer(self, input, weights, scope=None, name=None, share_trainables=True):
		layer_type = 'Weights'
		def layer_fn():
			kernel_layer = tf.keras.layers.Dense(name='TS_Dense0', units=1, activation=tf.nn.crelu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))
			weight_layer = tf.keras.layers.Dense(name='TS_Dense1', units=64, activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))
			input_layer = tf.keras.layers.Flatten()
			def exec_fn(i, w):
				kernel = tf.stop_gradient(w['kernel'])
				kernel = tf.transpose(kernel, [1, 0])
				kernel = kernel_layer(kernel)
				kernel = tf.reshape(kernel, [1,-1])
				bias = tf.stop_gradient(w['bias'])
				bias = tf.reshape(bias, [1,-1])
				weight_state = tf.concat((kernel, bias), -1)
				weight_state = weight_layer(weight_state)
				weight_state = tf.reshape(weight_state, [-1])
				i = input_layer(i)
				i = tf.map_fn(fn=lambda b: tf.concat((b,weight_state),-1), elems=i)
				return i
			return exec_fn
		return self._scopefy(inputs=(input, weights), output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)
