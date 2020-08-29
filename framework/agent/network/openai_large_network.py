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
		layer_type = 'Concat'
		def layer_fn():
			input_layer = tf.keras.layers.Flatten()
			concat_layer = tf.keras.layers.Flatten()
			final_layer = tf.keras.Sequential(name=layer_type, layers=[
				tf.keras.layers.Dense(name='Concat_Dense1',  units=256, activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2))),
				tf.keras.layers.Dense(name='Concat_Dense2',  units=448, activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2))),
			])
			def exec_fn(i,c):
				i = input_layer(i)
				if c.get_shape()[-1] > 0:
					c = concat_layer(c)
					i = tf.concat([i, c], -1) # shape: (batch, concat_size+units)
				return final_layer(i)
			return exec_fn
		return self._scopefy(inputs=(input, concat), output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def _rnn_layer(self, input, size_batch, scope="", name="", share_trainables=True):
		layer_type = rnn.type
		def layer_fn():
			rnn = RNN(type='LSTM', direction=1, units=448, batch_size=1, stack_size=1, training=self.training, dtype=flags.parameters_type)
			internal_initial_state = rnn.state_placeholder(name="initial_lstm_state") # for stateful lstm
			internal_default_state = rnn.default_state()
			def exec_fn(i, s):
				output, internal_final_state = rnn.process_batches(
					input=i, 
					initial_states=internal_initial_state, 
					sizes=s
				)
				return output, ([internal_initial_state],[internal_default_state],[internal_final_state])
			return exec_fn
		return self._scopefy(inputs=(input, size_batch), output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)
