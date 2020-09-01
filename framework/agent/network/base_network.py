# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf

import numpy as np
from agent.network.network import Network
import utils.tensorflow_utils as tf_utils
# from utils.rnn import RNN
import options
flags = options.get()

def is_continuous_control(policy_depth):
	return policy_depth <= 1

class Base_Network(Network):
	produce_explicit_relations = False

	def __init__(self, id, policy_heads, scope_dict, training=True, value_count=1, state_scaler=1):
		super().__init__(id, training)
		self.value_count = value_count
		# scope names
		self.scope_name = scope_dict['self']
		self.parent_scope_name = scope_dict['parent']
		self.sibling_scope_name = scope_dict['sibling']
		self.parameters_type = eval('tf.{}'.format(flags.parameters_type))
		self.policy_heads = policy_heads
		self.state_scaler = state_scaler
		
	def build_embedding(self, batch_dict, use_internal_state=True, scope=""):
		self.use_internal_state = use_internal_state
		print( "	[{}]Building partition {} use_internal_state={}".format(self.id, scope, use_internal_state) )
		print( "	[{}]Parameters type: {}".format(self.id, flags.parameters_type) )
		print( "	[{}]Algorithm: {}".format(self.id, flags.algorithm) )
		print( "	[{}]Network configuration: {}".format(self.id, flags.network_configuration) )
		# [Input]
		state_batch = [s for s in batch_dict['state'] if len(s.get_shape())==4]
		concat_batch = [s for s in batch_dict['state'] if len(s.get_shape())!=4]
		size_batch = batch_dict['size']
		environment_model = batch_dict['training_state'] if flags.use_learnt_environment_model_as_observation else None
		# [State Embedding]
		embedded_input = self._state_embedding_layer(
			state_batch, 
			concat_batch, 
			environment_model, 
			scope=self.format_scope_name([self.parent_scope_name,scope])
		)
		print( "	[{}]State Embedding shape: {}".format(self.id, embedded_input.get_shape()) )
		# [RNN]
		if use_internal_state:
			embedded_input, internal_state_tuple = self._rnn_layer(
				input=embedded_input, 
				size_batch=size_batch, 
				scope=self.format_scope_name([self.scope_name,scope])
			)
			self.internal_initial_state, self.internal_default_state, self.internal_final_state = internal_state_tuple
			print( "	[{}]RNN layer output shape: {}".format(self.id, embedded_input.get_shape()) )
			for i,h in enumerate(self.internal_initial_state):
				print( "	[{}]RNN{} initial state shape: {}".format(self.id, i, h.get_shape()) )
			for i,h in enumerate(self.internal_final_state):
				print( "	[{}]RNN{} final state shape: {}".format(self.id, i, h.get_shape()) )
		return embedded_input

	def _state_embedding_layer(self, state_batch, concat_batch, environment_model, scope="", name="", share_trainables=True):
		# [CNN]
		embedded_input = [
			self._cnn_layer(
				name=i, 
				input=substate_batch/self.state_scaler, 
				share_trainables=share_trainables, 
				scope=scope
			)
			for i,substate_batch in enumerate(state_batch)
		]
		embedded_input = tf.concat(embedded_input, 1)
		embedded_input = tf.keras.layers.Flatten()(embedded_input)
		print( "	[{}]CNN layer output shape: {}".format(self.id, embedded_input.get_shape()) )
		# [Training state]
		if flags.use_learnt_environment_model_as_observation:
			embedded_input = self._weights_layer(
				input=embedded_input, 
				weights=environment_model, 
				share_trainables=share_trainables, 
				scope=scope
			)
			print( "	[{}]Weights layer output shape: {}".format(self.id, embedded_input.get_shape()) )
		# [Concat]
		if len(concat_batch) > 0:
			embedded_input = self._concat_layer(
				input=embedded_input, 
				concat=tf.concat(list(map(tf.keras.layers.Flatten(), concat_batch)), -1), 
				share_trainables=share_trainables, 
				scope=scope,
			)
			print( "	[{}]Concat layer output shape: {}".format(self.id, embedded_input.get_shape()) )
		# embedded_input = tf.keras.layers.LayerNormalization(name='EmbeddingNorm')(embedded_input)
		return embedded_input

	def _cnn_layer(self, input, scope="", name="", share_trainables=True):
		layer_type = 'CNN'
		def layer_fn():
			input_layer = tf.keras.Sequential(name=layer_type, layers=[
				tf.keras.layers.Conv2D(name='CNN_Conv1',  filters=32, kernel_size=8, strides=4, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2))),
				tf.keras.layers.Conv2D(name='CNN_Conv2',  filters=64, kernel_size=4, strides=2, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2))),
				tf.keras.layers.Conv2D(name='CNN_Conv3',  filters=64, kernel_size=4, strides=1, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2))),
			])
			def exec_fn(i):
				return input_layer(i)
			return exec_fn
		return self._scopefy(inputs=(input, ), output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def _weights_layer(self, input, weights, scope="", name="", share_trainables=True):
		layer_type = 'Weights'
		def layer_fn():
			kernel_layer = tf.keras.layers.Dense(name='Concat_Dense0',  units=1, activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling)
			input_layer = tf.keras.layers.Flatten()
			def exec_fn(i, w):
				kernel = tf.stop_gradient(w['kernel'])
				kernel = tf.transpose(kernel, [1, 0])
				kernel = kernel_layer(kernel)
				kernel = tf.reshape(kernel, [1,-1])
				bias = tf.stop_gradient(w['bias'])
				bias = tf.reshape(bias, [1,-1])
				weight_state = tf.concat((kernel, bias), -1)
				weight_state = tf.reshape(weight_state, [-1])
				i = input_layer(i)
				i = tf.map_fn(fn=lambda b: tf.concat((b,weight_state),-1), elems=i)
				return i
			return exec_fn
		return self._scopefy(inputs=(input, weights), output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def _concat_layer(self, input, concat, scope, name="", share_trainables=True):
		layer_type = 'Concat'
		def layer_fn():
			input_layer = tf.keras.Sequential([
				tf.keras.layers.Flatten(),
				tf.keras.layers.Dense(name='Concat_Dense1',  units=64, activation=tf.nn.elu, kernel_initializer=tf.initializers.variance_scaling),
			])
			concat_layer = tf.keras.layers.Flatten()
			def exec_fn(i,c):
				i = input_layer(i)
				if c.get_shape()[-1] > 0:
					c = concat_layer(c)
					i = tf.concat([i, c], -1) # shape: (batch, concat_size+units)
				return i
			return exec_fn
		return self._scopefy(inputs=(input, concat), output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def value_layer(self, input, scope="", name="", share_trainables=True, qvalue_estimation=False):
		layer_type = 'Value'

		policy_depth = sum(h['depth'] for h in self.policy_heads)
		policy_size = sum(h['size'] for h in self.policy_heads)
		units = policy_size*max(1,policy_depth)
		def layer_fn():
			q_value_layers = [
				tf.keras.layers.Dense(name='{}_Q{}_Dense1'.format(layer_type,i),  units=units, activation=None, kernel_initializer=tf.initializers.variance_scaling)
				for i in range(self.value_count)
			]
			v_value_layers = [ # Keep value heads separated
				tf.keras.layers.Dense(name='{}_V{}_Dense1'.format(layer_type,i),  units=1, activation=None, kernel_initializer=tf.initializers.variance_scaling)
				for i in range(self.value_count)
			]
			v_output_layer = tf.keras.layers.Flatten()
			def exec_fn(i):
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
			policy_layer = [
				(
					tf.keras.layers.Dense(name='{}_Mu_Dense{}'.format(layer_type,h),  units=policy_head['size'], activation=None, kernel_initializer=tf.initializers.variance_scaling),
					tf.keras.layers.Dense(name='{}_Sigma_Dense{}'.format(layer_type,h),  units=policy_head['size'], activation=None, kernel_initializer=tf.initializers.variance_scaling)
				)
				if is_continuous_control(policy_head['depth']) else
				tf.keras.layers.Dense(name='{}_Logits_Dense{}'.format(layer_type,h),  units=policy_head['size']*policy_head['depth'], activation=None, kernel_initializer=tf.initializers.variance_scaling)
				for h,policy_head in enumerate(self.policy_heads)
			]
			def exec_fn(i):
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

	def _transition_prediction_layer(self, state, action, scope="", name="", share_trainables=True):
		layer_type = 'TransitionPredictor'
		def layer_fn():
			new_transition_layer = tf.keras.layers.Dense(
				name='{}_Dense_State'.format(layer_type), 
				units=xx.get_shape().as_list()[-1], 
				activation=None, 
				kernel_initializer=tf.initializers.variance_scaling
			)
			reward_layer = tf.keras.layers.Dense(
				name='{}_Dense_Reward'.format(layer_type), 
				units=1, 
				activation=None, 
				kernel_initializer=tf.initializers.variance_scaling
			)
			def exec_fn(s,a):
				s = tf.stop_gradient(s)
				state_action = tf.concat([tf.concat(a, -1),s], -1)
				new_transition_prediction = new_transition_layer(state_action)
				reward_prediction = new_transition_layer(state_action)
				return new_transition_prediction, reward_prediction
			return exec_fn
		return self._scopefy(inputs=(state, action), output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def _rnn_layer(self, input, size_batch, scope="", name="", share_trainables=True):
		layer_type = rnn.type
		def layer_fn():
			rnn = RNN(type='LSTM', direction=1, units=64, batch_size=1, stack_size=1, training=self.training, dtype=flags.parameters_type)
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
