# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.compat.v1 as tf
from agent.network.base_network import Base_Network, is_continuous_control
import utils.tensorflow_utils as tf_utils
from utils.rnn import RNN
from agent.network.openai_small_network import OpenAISmall_Network
import options
flags = options.get()

# N.B. tf.initializers.orthogonal is broken with tensorflow 1.10 and GPU, use OpenAI implementation
class MSA_Network(OpenAISmall_Network):

	# ===========================================================================
	# def _cnn_layer(self, input, scope, name="", share_trainables=True):
	# 	layer_type = 'CNN'
	# 	_, input_height, input_width, input_channel = input.get_shape().as_list()
	# 	def layer_fn():
	# 		xx = tf.keras.layers.Conv2D(name='CNN_Conv1',  filters=16, kernel_size=(input_height,1), dilation_rate=(1,3), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))(input)
	# 		xx = tf.keras.layers.Conv2D(name='CNN_Conv2',  filters=16, kernel_size=(1,input_width), dilation_rate=(3,1), padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))(xx)
	# 		xx = tf.keras.layers.Conv2D(name='CNN_Conv3',  filters=32, kernel_size=3, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))(xx)
	# 		return xx
	# 	return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)
	# ===========================================================================
		
	def _concat_layer(self, input, concat, scope, name="", share_trainables=True):
		layer_type = 'Concat'
		def layer_fn():
			xx = tf.keras.layers.Flatten()(input)
			concat = tf.keras.layers.Flatten()(concat)
			xx = tf.concat([xx, concat], -1) # shape: (batch, concat_size+input_size)
			xx = tf.keras.layers.Dense(name='Concat_Dense1',  units=256, activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))(xx)
			return xx
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def _rnn_layer(self, input, size_batch, scope, name="", share_trainables=True):
		rnn = RNN(type='LSTM', direction=2, units=128, batch_size=1, stack_size=1, training=self.training, dtype=flags.parameters_type)
		internal_initial_state = rnn.state_placeholder(name="initial_lstm_state") # for stateful lstm
		internal_default_state = rnn.default_state()
		layer_type = rnn.type
		def layer_fn():
			output, internal_final_state = rnn.process_batches(
				input=input, 
				initial_states=internal_initial_state, 
				sizes=size_batch
			)
			output = tf.layers.dropout(output, rate=0.75, training=self.training)
			return output, ([internal_initial_state],[internal_default_state],[internal_final_state])
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)
