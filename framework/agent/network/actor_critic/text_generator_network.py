# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.keras.layers as ktf
from agent.network.actor_critic.openai_small_network import OpenAISmall_Network
import utils.tensorflow_utils as tf_utils
from utils.rnn import RNN
import options
flags = options.get()

# N.B. tf.initializers.orthogonal is broken with tensorflow 1.10 and GPU, use OpenAI implementation

class TextGenerator_Network(OpenAISmall_Network):

	def _cnn_layer(self, input, scope, name="", share_trainables=True):
		layer_type = 'CNN'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			# input = tf.contrib.model_pruning.masked_conv2d(inputs=input, num_outputs=16, kernel_size=(3,3), padding='SAME', activation_fn=tf.nn.leaky_relu) # xavier initializer
			# input = tf.contrib.model_pruning.masked_conv2d(inputs=input, num_outputs=32, kernel_size=(3,3), padding='SAME', activation_fn=tf.nn.leaky_relu) # xavier initializer
			input = ktf.Conv2D(name='CNN_Conv0', filters=32, kernel_size=8, strides=8, padding='same', activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2))).apply(inputs=input)
			input = ktf.Conv2D(name='CNN_Conv1', filters=32, kernel_size=8, strides=4, padding='same', activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2))).apply(inputs=input)
			input = ktf.Conv2D(name='CNN_Conv2', filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2))).apply(inputs=input)
			input = ktf.Conv2D(name='CNN_Conv3', filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2))).apply(inputs=input)
			# update keys
			self._update_keys(variable_scope.name, share_trainables)
			# return result
			return input
