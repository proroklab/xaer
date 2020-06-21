# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
from agent.network.base_network import Base_Network

class HybridTowers_Network(Base_Network):
	
	# relu vs leaky_relu <https://www.reddit.com/r/MachineLearning/comments/4znzvo/what_are_the_advantages_of_relu_over_the/>
	def _cnn_layer(self, input, scope, name="", share_trainables=True):
		input_shape = input.get_shape().as_list()
		layer_type = 'CNN'
		def layer_fn():
			tower1 = tf.keras.layers.Conv2D(name='CNN_Tower1_Conv1',  filters=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling)(input)
			tower1 = tf.keras.layers.Conv2D(name='CNN_Tower1_Conv2',  filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling)(tower1)
			tower1 = tf.layers.max_pooling2d(tower1, pool_size=(input_shape[1], input_shape[2]), strides=(input_shape[1], input_shape[2]))
			tower1 = tf.layers.flatten(tower1)
			xx = tf.keras.layers.Conv2D(name='CNN_Tower2_Conv1',  filters=16, kernel_size=(3,3), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling )(input)
			xx = tf.keras.layers.Conv2D(name='CNN_Tower2_Conv2',  filters=8, kernel_size=(3,3), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling )(xx)
			xx = tf.layers.flatten(xx)
			concat = tf.concat([tower1, xx], axis=-1)
			return concat
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)
