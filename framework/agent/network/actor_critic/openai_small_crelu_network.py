# -*- coding: utf-8 -*-
import options
flags = options.get()

import numpy as np
import tensorflow.compat.v1 as tf
from agent.network.actor_critic.openai_small_network import OpenAISmall_Network
import utils.tensorflow_utils as tf_utils
from utils.rnn import RNN

# N.B. tf.initializers.orthogonal is broken with tensorflow 1.10 and GPU, use OpenAI implementation

class OpenAISmallCRELU_Network(OpenAISmall_Network):

	def _cnn_layer(self, input, scope, name="", share_trainables=True):
		layer_type = 'CNN'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			# input = tf.contrib.model_pruning.masked_conv2d(inputs=input, num_outputs=16, kernel_size=(3,3), padding='SAME', activation_fn=tf.nn.leaky_relu) # xavier initializer
			# input = tf.contrib.model_pruning.masked_conv2d(inputs=input, num_outputs=32, kernel_size=(3,3), padding='SAME', activation_fn=tf.nn.leaky_relu) # xavier initializer
			input = tf.layers.conv2d(name='CNN_Conv1', inputs=input, filters=32, kernel_size=8, strides=4, padding='SAME', activation=tf.nn.crelu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))
			input = tf.layers.conv2d(name='CNN_Conv2', inputs=input, filters=64, kernel_size=4, strides=2, padding='SAME', activation=tf.nn.crelu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))
			input = tf.layers.conv2d(name='CNN_Conv3', inputs=input, filters=64, kernel_size=4, strides=1, padding='SAME', activation=tf.nn.crelu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))
			# update keys
			self._update_keys(variable_scope.name, share_trainables)
			# return result
			return input
		
	def _weights_layer(self, input, weights, scope, name="", share_trainables=True):
		layer_type = 'Weights'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			kernel = tf.stop_gradient(weights['kernel'])
			kernel = tf.transpose(kernel, [1, 0])
			kernel = tf.layers.dense(name='TS_Dense0', inputs=kernel, units=1, activation=tf.nn.crelu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))
			kernel = tf.reshape(kernel, [1,-1])
			bias = tf.stop_gradient(weights['bias'])
			bias = tf.reshape(bias, [1,-1])
			weight_state = tf.concat((kernel, bias), -1)
			weight_state = tf.layers.dense(name='TS_Dense1', inputs=weight_state, units=64, activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))
			weight_state = tf.reshape(weight_state, [-1])
			input = tf.layers.flatten(input)
			input = tf.map_fn(fn=lambda b: tf.concat((b,weight_state),-1), elems=input)
			# Update keys
			self._update_keys(variable_scope.name, share_trainables)
			# Return result
			return input
