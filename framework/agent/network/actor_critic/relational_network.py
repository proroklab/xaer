# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.compat.v1 as tf
from agent.network.actor_critic.openai_small_network import OpenAISmall_Network
import utils.tensorflow_utils as tf_utils
import options
flags = options.get()

class Relational_Network(OpenAISmall_Network):

	@staticmethod
	def entity_extraction_layer(x):
		coordinates = Relational_Network.coordinates_layer(x)
		# [B,Height,W,D+2]
		x = tf.concat([x, coordinates], axis=3)
		# [B,Height,W,D]
		x = tf.keras.layers.Conv2D(name='CNN_Conv1', filters=32, kernel_size=8, strides=1, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))(x)
		x = tf.keras.layers.Conv2D(name='CNN_Conv2', filters=64, kernel_size=4, strides=1, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))(x)
		x = tf.keras.layers.Conv2D(name='CNN_Conv3', filters=64, kernel_size=4, strides=1, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))(x)
		# [B,N,D] N=Height*w
		_, h, w, channels = x.shape.as_list()
		entities = tf.reshape(x, [-1, h * w, channels])
		return entities

	@staticmethod
	def predicate_invention_layer(entities):
		query_key_value = Relational_Network.query_key_value_layer(entities)
		query, key, value = map(tf.keras.layers.LayerNormalization(center=True, scale=True, trainable=False), query_key_value)
		accumulated_interaction, attention_weight = Relational_Network.attention_layer(query, key, value)
		# print( "	Accumulated interactions shape: {}".format(accumulated_interaction.get_shape()) )
		# print( "	Attention weights shape: {}".format(attention_weight.get_shape()) )
		invented_predicate = Relational_Network.residual_block(entities, accumulated_interaction)
		invented_predicate = tf.keras.layers.LayerNormalization(center=True, scale=True, trainable=False)(invented_predicate)
		return invented_predicate, attention_weight

	def _relational_layer(self, input, scope, name="", share_trainables=True):
		layer_type = 'RelationalNet'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			entities = self.entity_extraction_layer(input)
			print( "	Entity Extraction layer output shape: {}".format(entities.get_shape()) )
			invented_predicates, attention_weight = self.predicate_invention_layer(entities)
			print( "	Predicate Invention layer shape: {}".format(invented_predicates.get_shape()) )
			input = tf.reduce_max(invented_predicates, axis=-1) # feature_wise_max
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

	@staticmethod
	def query_key_value_layer(entities, n_heads=2):
		"""
		:param entities: (TensorFlow Tensor) entities [B,N,D]
		:param n_heads: (float) The number of attention heads to use
		:return: (TensorFlow Tensor) [B,n_heads,N,D]
		"""
		query_size = key_size = value_size = entities.shape[-1].value
		embedding_sizes = [query_size, key_size, value_size]
		N = entities.shape[1].value
		channels = entities.shape[2].value
		# total_size Denoted as F, n_heads Denoted as H
		total_size = sum(embedding_sizes) * n_heads
		# [B*N,D]
		entities = tf.reshape(entities, [-1, channels])
		# [B*N,F] F = sum(embedding_sizes) * n_heads
		embedded_entities = tf.keras.layers.Dense(units=total_size, activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))(entities)
		# [B*N,F] --> [B,N,F] new
		qkv = tf.reshape(embedded_entities, [-1, N, total_size])
		# [B*N,F]
		# qkv = tf.keras.layers.LayerNormalization(center=True, scale=True, trainable=False)(embedded_entities)
		# [B,N,n_heads,sum(embedding_sizes)]
		qkv = tf.reshape(qkv, [-1, N, n_heads, sum(embedding_sizes)])
		# [B,N,n_heads,sum(embedding_sizes)] -> [B,n_heads,N,sum(embedding_sizes)]
		qkv = tf.transpose(qkv, [0, 2, 1, 3])
		return tf.split(qkv, embedding_sizes, -1)

	@staticmethod
	def attention_layer(query, key, value):
		# [B,n_heads,N,N]
		query_key_similarity = tf.matmul(query, key, transpose_b=True)
		# [B,n_heads,N1,N2]
		channels = value.shape[-1].value
		attention_weight = tf.nn.softmax(query_key_similarity * (channels**-0.5))
		# [B,n_heads,N1,D]
		accumulated_interaction = tf.matmul(attention_weight, value)
		return accumulated_interaction, attention_weight

	@staticmethod
	def coordinates_layer(input_tensor):
		"""
		The output of cnn is tagged with two extra channels indicating the spatial position(x and y) of each cell

		:param input_tensor: (TensorFlow Tensor)  [B,Height,W,D]
		:return: (TensorFlow Tensor) [B,Height,W,2]
		"""
		batch_size = tf.shape(input_tensor)[0]
		_, height, width, _ = input_tensor.shape.as_list()
		coor = [[[h / height, w / width] for w in range(width)] for h in range(height)]
		# coor = []
		# for h in range(height):
		#	 w_channel = []
		#	 for w in range(width):
		#		 w_channel.append([float(h / height), float(w / width)])
		#	 coor.append(w_channel)
		coor = tf.expand_dims(tf.constant(coor, dtype=input_tensor.dtype), axis=0)
		coor = tf.convert_to_tensor(coor)
		# [1,Height,W,2] --> [B,Height,W,2]
		coor = tf.tile(coor, [batch_size, 1, 1, 1])
		return coor

	@staticmethod
	def residual_block(x, y):
		"""
		Z = W*y + x
		:param x: (TensorFlow Tensor) entities [B,N,D] N = n_entities
		:param y: (TensorFlow Tensor) new_entities from MHDPA [B,n_heads,N,D]
		:return: (TensorFlow Tensor) [B,n_heads,N,D]
		"""
		x = tf.expand_dims(x, axis=1)
		# [B,1,N,D] --> [B,n_heads,N,D]
		x = tf.tile(x, [1,  y.shape[1].value, 1, 1])
		y = tf.keras.layers.Dense(units=y.shape[3].value, activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))(y)
		return tf.add(y, x)
	