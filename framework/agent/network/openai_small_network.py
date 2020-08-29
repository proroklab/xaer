# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.compat.v1 as tf

from agent.network.base_network import Base_Network, is_continuous_control
import utils.tensorflow_utils as tf_utils
from utils.rnn import RNN
import options
flags = options.get()

# N.B. tf.initializers.orthogonal is broken with tensorflow 1.10 and GPU, use OpenAI implementation

class OpenAISmall_Network(Base_Network):

	def _concat_layer(self, input, concat, scope="", name="", share_trainables=True):
		layer_type = 'Concat'
		def layer_fn():
			input_layer = tf.keras.layers.Flatten()
			concat_layer = tf.keras.layers.Flatten()
			final_layer = tf.keras.layers.Dense(name='Concat_Dense1',  units=256, activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))
			def exec_fn(i,c):
				i = input_layer(i)
				if c.get_shape()[-1] > 0:
					c = concat_layer(c)
					i = tf.concat([i, c], -1) # shape: (batch, concat_size+units)
				return final_layer(i)
			return exec_fn
		return self._scopefy(inputs=(input, concat), output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def _weights_layer(self, input, weights, scope="", name="", share_trainables=True):
		layer_type = 'Weights'
		def layer_fn():
			kernel_layer = tf.keras.layers.Dense(name='TS_Dense0', units=1, activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))
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

	def value_layer(self, input, scope="", name="", share_trainables=True, qvalue_estimation=False):
		layer_type = 'Value'

		policy_depth = sum(h['depth'] for h in self.policy_heads)
		policy_size = sum(h['size'] for h in self.policy_heads)
		units = policy_size*max(1,policy_depth)
		def layer_fn():
			input_layer = tf.keras.layers.Dense(name='Value_Dense1', units=input.get_shape().as_list()[-1], activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(0.1))
			q_value_layers = [
				tf.keras.layers.Dense(name='Value_Q{}_Dense1'.format(i), units=units, activation=None, kernel_initializer=tf_utils.orthogonal_initializer(0.01))
				for i in range(self.value_count)
			]
			v_value_layers = [ # Keep value heads separated
				tf.keras.layers.Dense(name='Value_V{}_Dense1'.format(i),  units=1, activation=None, kernel_initializer=tf_utils.orthogonal_initializer(0.01))
				for i in range(self.value_count)
			]
			v_output_layer = tf.keras.layers.Flatten()
			def exec_fn(i):
				i = i + input_layer(i)
				if qvalue_estimation:
					i = [l(i) for l in q_value_layers]
					i = tf.stack(i)
					i = tf.transpose(i, [1, 0, 2])
					if policy_size > 1 and policy_depth > 1:
						i = tf.reshape(i, [-1,self.value_count,policy_size,policy_depth])
				else:
					i = [l(i) for l in v_value_layers]
					i = tf.stack(i)
					i = tf.transpose(i, [1, 0, 2])
					i = v_output_layer(i)
				return i
			return exec_fn
		return self._scopefy(inputs=(input, ), output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def policy_layer(self, input, scope="", name="", share_trainables=True):
		layer_type = 'Policy'
		def layer_fn():
			input_layer = tf.keras.layers.Dense(name='Policy_Dense1',  units=input.get_shape().as_list()[-1], activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(0.1))
			policy_layer = [
				(
					tf.keras.layers.Dense(name='Policy_Mu_Dense{}'.format(h),  units=policy_head['size'], activation=None, kernel_initializer=tf_utils.orthogonal_initializer(0.01)),
					tf.keras.layers.Dense(name='Policy_Sigma_Dense{}'.format(h),  units=policy_head['size'], activation=None, kernel_initializer=tf_utils.orthogonal_initializer(0.01))
				)
				if is_continuous_control(policy_head['depth']) else
				tf.keras.layers.Dense(name='Policy_Logits_Dense{}'.format(h),  units=policy_head['size']*policy_head['depth'], activation=None, kernel_initializer=tf_utils.orthogonal_initializer(0.01))
				for h,policy_head in enumerate(self.policy_heads)
			]
			def exec_fn(i):
				i = i + input_layer(i)
				output_list = []
				for h,policy_head in enumerate(self.policy_heads):
					policy_depth = policy_head['depth']
					policy_size = policy_head['size']
					if is_continuous_control(policy_depth):
						# build mean
						mu = policy_layer[h][0](i) # in (-inf,inf)
						# build standard deviation
						sigma = policy_layer[h][1](i) # in (-inf,inf)
						# clip mu and sigma to avoid numerical instabilities
						clipped_mu = tf.clip_by_value(mu, -1,1, name='mu_clipper') # in [-1,1]
						clipped_sigma = tf.clip_by_value(tf.abs(sigma), 1e-4,1, name='sigma_clipper') # in [1e-4,1] # sigma must be greater than 0
						# build policy batch
						policy_batch = tf.stack([clipped_mu, clipped_sigma])
						policy_batch = tf.transpose(policy_batch, [1, 0, 2])
					else: # discrete control
						policy_batch = policy_layer[h](i)
						if policy_size > 1:
							policy_batch = tf.reshape(policy_batch, [-1,policy_size,policy_depth])
					output_list.append(policy_batch)
				return output_list
			return exec_fn
		return self._scopefy(inputs=(input, ), output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def _rnn_layer(self, input, size_batch, scope="", name="", share_trainables=True):
		layer_type = rnn.type
		def layer_fn():
			rnn = RNN(type='GRU', direction=1, units=256, batch_size=1, stack_size=1, training=self.training, dtype=flags.parameters_type)
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
