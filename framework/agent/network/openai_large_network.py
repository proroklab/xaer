# -*- coding: utf-8 -*-
import options
flags = options.get()

import numpy as np
import tensorflow.compat.v1 as tf
from agent.network.openai_small_network import OpenAISmall_Network
import utils.tensorflow_utils as tf_utils
from utils.rnn import RNN

# N.B. tf.initializers.orthogonal is broken with tensorflow 1.10 and GPU, use OpenAI implementation

class OpenAILarge_Network(OpenAISmall_Network):

	def _concat_layer(self, input, concat, scope, name="", share_trainables=True):
		def layer_fn():
			xx = tf.layers.flatten(input)
			if concat.get_shape()[-1] > 0:
				concat = tf.layers.flatten(concat)
				xx = tf.concat([xx, concat], -1) # shape: (batch, concat_size+units)
			xx = tf.keras.layers.Dense(name='Concat_Dense1',  units=256, activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))(xx)
			xx = tf.keras.layers.Dense(name='Concat_Dense2',  units=448, activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))(xx)
			return xx
		return self._scopefy(output_fn=layer_fn, layer_type='Concat', scope=scope, name=name, share_trainables=share_trainables)
		
	def _rnn_layer(self, input, size_batch, scope, name="", share_trainables=True):
		rnn = RNN(type='LSTM', direction=1, units=448, batch_size=1, stack_size=1, training=self.training, dtype=flags.parameters_type)
		internal_initial_state = rnn.state_placeholder(name="initial_lstm_state") # for stateful lstm
		internal_default_state = rnn.default_state()
		def layer_fn():
			output, internal_final_state = rnn.process_batches(
				input=input, 
				initial_states=internal_initial_state, 
				sizes=size_batch
			)
			return output, ([internal_initial_state],[internal_default_state],[internal_final_state])
		return self._scopefy(output_fn=layer_fn, layer_type=rnn.type, scope=scope, name=name, share_trainables=share_trainables)
		