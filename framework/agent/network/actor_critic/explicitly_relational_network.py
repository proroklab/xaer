# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.compat.v1 as tf
from agent.network.actor_critic.openai_small_network import OpenAISmall_Network
import options
flags = options.get()

HEADS = 2
RELATIONS_PER_HEAD = 16

# Shanahan, Murray, et al. "An explicitly relational neural network architecture." arXiv preprint arXiv:1905.10307 (2019).
class ExplicitlyRelational_Network(OpenAISmall_Network):
	kernel_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1)
	bias_initializer = tf.constant_initializer(0.1)

	def entity_extraction_layer(self, x, scope, name="", share_trainables=True):
		# [B,Height,W,D]
		x = self._cnn_layer(input=x, scope=scope, name=name, share_trainables=share_trainables)
		# [B,Height,W,D+2]
		coordinates = self.coordinates_layer(x)
		x = tf.concat([x, coordinates], axis=-1)
		# [B,N,D] N=Height*w
		_, h, w, channels = x.shape.as_list()
		entities = tf.reshape(x, [-1, h * w, channels])
		return entities

	def relation_extraction_layer(self, entities, n_relations, n_heads):
		keys = self.key_layer(entities, n_heads)
		query1 = self.query_layer(entities, n_heads)
		query2 = self.query_layer(entities, n_heads)
		# Attention weights
		keys_t = tf.transpose(keys, perm=[0, 1, 3, 2])
		# (batch_size, heads, channels, conv_out_size*conv_out_size)
		att1 = tf.nn.softmax(tf.matmul(query1, keys_t))
		att2 = tf.nn.softmax(tf.matmul(query2, keys_t))
		# (batch_size, heads, 1, conv_out_size*conv_out_size)

		# Reshape features
		features_tiled = tf.tile(tf.expand_dims(entities, 1), [1, n_heads, 1, 1])
		# (batch_size, heads, conv_out_size*conv_out_size, channels+2)

		# Compute a pair of features using attention weights
		left_objects = tf.squeeze(tf.matmul(att1, features_tiled),axis=-2)
		right_objects = tf.squeeze(tf.matmul(att2, features_tiled),axis=-2)
		# (batch_size, heads, (channels+2))

		# Spatial embedding
		left_objects_embedding = tf.keras.layers.Dense(
			units=n_relations, 
			# activation=tf.nn.relu, # non-linear mapping
			kernel_initializer=self.kernel_initializer,
			bias_initializer=self.bias_initializer,
			
		)(left_objects)
		right_objects_embedding = tf.keras.layers.Dense(
			units=n_relations, 
			# activation=tf.nn.relu, # non-linear mapping
			kernel_initializer=self.kernel_initializer,
			bias_initializer=self.bias_initializer,
		)(right_objects)
		# (batch_size, heads, relations)

		# Comparator
		relations = tf.subtract(left_objects_embedding, right_objects_embedding)

		# Positions
		channels = entities.shape.as_list()[-1] - 2 # x,y coordinates are at the end
		pos1 = tf.slice(left_objects, [0, 0, channels], [-1, -1, -1])
		pos2 = tf.slice(right_objects, [0, 0, channels], [-1, -1, -1])
		# (batch_size, heads, 2)

		# Collect relations and concatenate positions (objects)
		triples = tf.concat([relations, pos1, pos2], 2)
		# (batch_size, heads, relations+4)
		return triples

	def _relational_layer(self, input, scope, name="", share_trainables=True):
		layer_type = 'RelationalNet'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			entities = self.entity_extraction_layer(input, scope, name='EntityExtraction', share_trainables=share_trainables)
			print( "	Entity Extraction layer output shape: {}".format(entities.get_shape()) )
			relations = self.relation_extraction_layer(entities, n_relations=RELATIONS_PER_HEAD, n_heads=HEADS)
			relations = tf.layers.flatten(relations)
			print( "	Predicate Invention layer shape: {}".format(relations.get_shape()) )
			# update keys
			self._update_keys(variable_scope.name, share_trainables)
			# return result
			return relations
	
	def _state_embedding_layer(self, state_batch, concat_batch):
		embedded_input = [
			self._relational_layer(name=f'StateRelational{i}', input=substate_batch/self.state_scaler, scope=self.parent_scope_name)
			for i,substate_batch in enumerate(state_batch)
		]
		embedded_input = [
			tf.layers.flatten(i)
			for i in embedded_input
		]
		embedded_input = tf.concat(embedded_input, -1)
		print( "	[{}]State Relational layer output shape: {}".format(self.id, embedded_input.get_shape()) )
		# [Concat]
		if len(concat_batch) > 0:
			concat_batch = tf.concat(concat_batch, -1)
			embedded_input = self._concat_layer(input=embedded_input, concat=concat_batch, scope=self.parent_scope_name)
			print( "	[{}]Concat layer output shape: {}".format(self.id, embedded_input.get_shape()) )
			# embedded_input = self._relational_layer(name=f'ConcatRelational', input=embedded_input, scope=self.parent_scope_name)
		# [Training state]
		if flags.use_learnt_environment_model_as_observation:
			embedded_input = self._weights_layer(input=embedded_input, weights=self.training_state, scope=self.scope_name)
			print( "	[{}]Weights layer output shape: {}".format(self.id, embedded_input.get_shape()) )
		# attention = np.stack(attention)[0]
		# attention = np.sum(attention, axis=0)
		# attention = np.reshape(attention, [int(np.sqrt(attention.shape[0])), int(np.sqrt(attention.shape[0]))])
		# plt.imshow(attention, cmap='gray')
		# plt.show(block=False)
		# plt.pause(0.001)
		# plt.clf()
		return embedded_input

	def _concat_layer(self, input, concat, scope, name="", share_trainables=True):
		layer_type = 'Concat'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			input = tf.layers.flatten(input)
			# input = tf.keras.layers.Dense(name='Concat_Dense1',  units=64, activation=tf.nn.elu, kernel_initializer=tf.initializers.variance_scaling)(input)
			if concat.get_shape()[-1] > 0:
				concat = tf.layers.flatten(concat)
				input = tf.concat([input, concat], -1) # shape: (batch, concat_size+units)
			# Update keys
			self._update_keys(variable_scope.name, share_trainables)
			# Return result
			return input

	def key_layer(self, entities, n_heads):
		key_size = entities.shape.as_list()[-1]
		# Keys
		key = tf.keras.layers.Dense( # linear mapping
			units=key_size, 
			activation=tf.nn.relu, 
			kernel_initializer=self.kernel_initializer,
			bias_initializer=self.bias_initializer,
		)(entities)
		return tf.tile(tf.expand_dims(key, 1), [1, n_heads, 1, 1])

	def query_layer(self, entities, n_heads):
		key_size = entities.shape.as_list()[-1]
		# Queries
		flatten_entities = tf.layers.flatten(entities)
		
		query = tf.keras.layers.Dense( # linear mapping
			units=n_heads*key_size, 
			activation=tf.nn.relu, 
			kernel_initializer=self.kernel_initializer,
			bias_initializer=self.bias_initializer,
		)(flatten_entities)
		# (batch_size, heads*key_size)
		query = tf.reshape(query, [-1, n_heads, key_size])
		# (batch_size, heads, key_size)
		query = tf.expand_dims(query, 2)
		# (batch_size, heads, 1, key_size)
		return query

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
