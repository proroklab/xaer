# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.compat.v1 as tf
from agent.network.actor_critic.base_network import Base_Network, is_continuous_control
from agent.network.actor_critic.openai_small_network import OpenAISmall_Network
import utils.tensorflow_utils as tf_utils
from utils.rnn import RNN
from utils.wavenet import WaveNetModel
import options
flags = options.get()

# N.B. tf.initializers.orthogonal is broken with tensorflow 1.10 and GPU, use OpenAI implementation
class Wave_Network(OpenAISmall_Network):
	wavenet_params = { # http://deepsound.io/wavenet_first_try.html
		"filter_width": 2,
		"residual_channels": 8,
		"dilation_channels": 8,
		"skip_channels": 128,
		"use_biases": True,
		"dilation_stack_count": 2,
	}

	def _cnn_layer(self, input, scope, name="", share_trainables=True):
		layer_type = 'WNN'
		_, input_height, input_width, input_channel = input.get_shape().as_list()
		input_size = input_height*input_width
		input_size_bits = int(np.ceil(np.log2(input_size)))
		# build dilations
		N = max(0, input_size_bits-1)
		M = self.wavenet_params['filter_width']
		dilations = [2**i for i in range(N)] * M
		# build layer
		def layer_fn():
			net = WaveNetModel(
				dilations=dilations,
				filter_width=self.wavenet_params['filter_width'],
				residual_channels=self.wavenet_params['residual_channels'],
				dilation_channels=self.wavenet_params['dilation_channels'],
				quantization_channels=input_channel,
				skip_channels=self.wavenet_params['skip_channels'],
				use_biases=self.wavenet_params['use_biases']
			)
			xx = tf.reshape(input, [-1, input_size, input_channel])
			xx = net.create_network(xx)
			return xx
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def _concat_layer(self, input, concat, scope, name="", share_trainables=True):
		layer_type = 'Concat'
		def layer_fn():
			xx = tf.layers.flatten(input)
			concat = tf.layers.flatten(concat)
			xx = tf.concat([xx, concat], -1) # shape: (batch, concat_size+input_size)
			#xx = tf.keras.layers.Dense(name='Concat_Dense1',  units=128, activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))(xx)
			return xx
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def _rnn_layer(self, input, scope, name="", share_trainables=True):
		rnn = RNN(type='LSTM', direction=2, units=128, batch_size=1, stack_size=1, training=self.training, dtype=flags.parameters_type)
		internal_initial_state = rnn.state_placeholder(name="initial_lstm_state") # for stateful lstm
		internal_default_state = rnn.default_state()
		layer_type = rnn.type
		def layer_fn():
			output, internal_final_state = rnn.process_batches(
				input=input, 
				initial_states=internal_initial_state, 
				sizes=self.size_batch
			)
			output = tf.layers.dropout(output, rate=0.5, training=self.training)
			return output, ([internal_initial_state],[internal_default_state],[internal_final_state])
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)
