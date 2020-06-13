# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
from agent.network.actor_critic.base_network import Base_Network
import options
flags = options.get()

def attention_CNN(x):
	x = tf.keras.layers.Conv2D(name='CNN_Conv1', filters=12, kernel_size=[3, 3], strides=1, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling)(x)
	x = tf.keras.layers.Conv2D(name='CNN_Conv2', filters=24, kernel_size=[3, 3], strides=1, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling)(x)
	shape = x.get_shape()
	print( "	Attention layer output shape: {}".format(shape) )
	return x, [s.value for s in shape]

def flatten(nnk, shape):
	flatten = tf.reshape(nnk, [-1, shape[1]*shape[2]*shape[3]])
	return flatten

def query_key_value(nnk, shape):
	flatten = tf.reshape(nnk, [-1, shape[1]*shape[2], shape[3]])
	after_layer = [tf.keras.layers.Dense(units=shape[3], activation=tf.nn.relu)(flatten) for i in range(3)]

	return after_layer[0], after_layer[1], after_layer[2], flatten

def self_attention(query, key, value):
	key_dim_size = float(key.get_shape().as_list()[-1])
	key = tf.transpose(key, perm=[0, 2, 1])
	S = tf.matmul(query, key) / tf.sqrt(key_dim_size)
	attention_weight = tf.nn.softmax(S)
	A = tf.matmul(attention_weight, value)
	shape = A.get_shape()
	print( "	Self Attention layer output shape: {}".format(shape) )
	return A, attention_weight, [s.value for s in shape]

def layer_normalization(x):
	feature_shape = x.get_shape()[-1:]
	mean, variance = tf.nn.moments(x, [2], keep_dims=True)
	beta = tf.Variable(tf.zeros(feature_shape), trainable=False)
	gamma = tf.Variable(tf.ones(feature_shape), trainable=False)
	return gamma * (x - mean) / tf.sqrt(variance + 1e-8) + beta

def residual(x, inp, residual_time):
	for i in range(residual_time):
		x = x + inp
		x = layer_normalization(x)
	return x

def feature_wise_max(x):
	return tf.reduce_max(x, axis=-1)

def relational_network(x):
	nnk, shape = attention_CNN(x)
	query, key, value, E = query_key_value(nnk, shape)
	normalized_query = layer_normalization(query)
	normalized_key = layer_normalization(key)
	normalized_value = layer_normalization(value)
	A, attention_weight, shape = self_attention(normalized_query, normalized_key, normalized_value)
	E_hat = residual(A, E, 2)
	print( "	Residual layer output shape: {}".format(E_hat.get_shape()) )
	max_E_hat = feature_wise_max(E_hat)
	# actor = output_layer(max_E_hat, hidden, output_size, activation, final_activation)
	# critic = tf.squeeze(output_layer(max_E_hat, hidden, 1, tf.nn.relu, None), axis=1)
	# return actor, critic, attention_weight
	return attention_weight, max_E_hat

class Relational_Network(Base_Network):

	def _relational_layer(self, input, scope, name="", share_trainables=True):
		layer_type = 'RelationalNet'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			attention_weight, input = relational_network(input)
			# update keys
			self._update_keys(variable_scope.name, share_trainables)
			# return result
			return attention_weight, input
	
	def _state_embedding_layer(self, state_batch, concat_batch):
		attention_weight_batch_list, embedded_input = zip(*[
			self._relational_layer(name=f'State{i}', input=substate_batch/self.state_scaler, scope=self.parent_scope_name)
			for i,substate_batch in enumerate(state_batch)
		])
		embedded_input = [
			tf.layers.flatten(i)
			for i in embedded_input
		]
		embedded_input = tf.concat(embedded_input, -1)
		print( "	[{}]Relational layer output shape: {}".format(self.id, embedded_input.get_shape()) )
		# [Training state]
		if flags.use_learnt_environment_model_as_observation:
			embedded_input = self._weights_layer(input=embedded_input, weights=self.training_state, scope=self.scope_name)
			print( "	[{}]Weights layer output shape: {}".format(self.id, embedded_input.get_shape()) )
		# [Concat]
		if len(concat_batch) > 0:
			concat_attention_weight_batch_list, concat_embedded_input = zip(*[
				self._relational_layer(name=f'Concat{i}', input=tf.expand_dims(tf.expand_dims(subconcat_batch,-1),-1), scope=self.parent_scope_name)
				for i,subconcat_batch in enumerate(concat_batch)
			])
			concat_embedded_input = [
				tf.layers.flatten(i)
				for i in concat_embedded_input
			]
			concat_embedded_input = tf.concat(concat_embedded_input, -1)

			attention_weight_batch_list += concat_attention_weight_batch_list
			embedded_input = tf.concat([embedded_input, concat_embedded_input], -1) # shape: (batch, concat_size+units)
			print( "	[{}]Concat layer output shape: {}".format(self.id, embedded_input.get_shape()) )
		self.attention_weight_batch_list = attention_weight_batch_list
		# attention = np.stack(attention)[0]
		# attention = np.sum(attention, axis=0)
		# attention = np.reshape(attention, [int(np.sqrt(attention.shape[0])), int(np.sqrt(attention.shape[0]))])
		# plt.imshow(attention, cmap='gray')
		# plt.show(block=False)
		# plt.pause(0.001)
		# plt.clf()
		return embedded_input
		