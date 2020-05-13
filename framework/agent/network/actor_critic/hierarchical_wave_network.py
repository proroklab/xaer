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
class HWave_Network(OpenAISmall_Network):
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
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			net = WaveNetModel(
				dilations=dilations,
				filter_width=self.wavenet_params['filter_width'],
				residual_channels=self.wavenet_params['residual_channels'],
				dilation_channels=self.wavenet_params['dilation_channels'],
				quantization_channels=input_channel,
				skip_channels=self.wavenet_params['skip_channels'],
				use_biases=self.wavenet_params['use_biases']
			)
			input = tf.reshape(input, [-1, input_size, input_channel])
			input = net.create_network(input)
			# update keys
			self._update_keys(variable_scope.name, share_trainables)
			# return result
			return input
		
	def _concat_layer(self, input, concat, scope, name="", share_trainables=True):
		layer_type = 'Concat'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			input = tf.layers.flatten(input)
			concat = tf.layers.flatten(concat)
			input = tf.concat([input, concat], -1) # shape: (batch, concat_size+input_size)
			#input = tf.layers.dense(name='Concat_Dense1', inputs=input, units=128, activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))
			# Update keys
			self._update_keys(variable_scope.name, share_trainables)
			# Return result
			return input
		
	def _rnn_layer(self, input, scope, name="", share_trainables=True):
		rnns = [
			RNN(type='LSTM', direction=2, units=128, batch_size=1, stack_size=1, training=self.training, dtype=flags.parameters_type),
			RNN(type='GRU', direction=1, units=128, batch_size=1, stack_size=1, training=self.training, dtype=flags.parameters_type)
		]
		internal_initial_state = [rnn.state_placeholder(name="initial_lstm_state") for rnn in rnns]
		internal_default_state = [rnn.default_state() for rnn in rnns]
		layer_type = '-'.join(rnn.type for rnn in rnns)
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			output_low, internal_final_state_low = rnns[0].process_batches(
				input=input, 
				initial_states=internal_initial_state[0], 
				sizes=self.size_batch
			)
			output_low = tf.layers.dropout(output_low, rate=0.5, training=self.training)
			output_high, internal_final_state_high = rnns[1].process_batches(
				input= internal_final_state_low, 
				initial_states= internal_initial_state[1], 
				sizes= 0*self.size_batch+1
			)
			output_high = tf.layers.dropout(output_high, rate=0.5, training=self.training)
			output = tf.concat((output_low,output_high),-1)
			internal_final_state = [internal_final_state_low, internal_final_state_high]
			# Update keys
			self._update_keys(variable_scope.name, share_trainables)
			return output, (internal_initial_state,internal_default_state,internal_final_state)
		