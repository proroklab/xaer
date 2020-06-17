# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.compat.v1 as tf
import utils.tensorflow_utils as tf_utils
from agent.network.actor_critic.openai_small_network import OpenAISmall_Network
import options
flags = options.get()

OBJECT_PAIRS = 16
EDGE_SIZE_PER_OBJECT_PAIR = 4

# Shanahan, Murray, et al. "An explicitly relational neural network architecture." arXiv preprint arXiv:1905.10307 (2019).
class ExplicitlyRelational_Network(OpenAISmall_Network):
	produce_explicit_relations = True
	kernel_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1)

	# def _cnn_layer(self, input, scope, name="", share_trainables=True):
	# 	layer_type = 'CNN'
	# 	def layer_fn():
	# 		# input = tf.contrib.model_pruning.masked_conv2d(inputs=input, num_outputs=16, kernel_size=(3,3), padding='SAME', activation_fn=tf.nn.leaky_relu) # xavier initializer
	# 		# input = tf.contrib.model_pruning.masked_conv2d(inputs=input, num_outputs=32, kernel_size=(3,3), padding='SAME', activation_fn=tf.nn.leaky_relu) # xavier initializer
	# 		xx = tf.keras.layers.Conv2D(name='CNN_Conv1',  filters=32, kernel_size=8, strides=4, padding='SAME', kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))(input)
	# 		xx = tf.keras.layers.Conv2D(name='CNN_Conv2',  filters=64, kernel_size=4, strides=2, padding='SAME', kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))(xx)
	# 		xx = tf.keras.layers.Conv2D(name='CNN_Conv3',  filters=64, kernel_size=4, strides=1, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))(xx)
	# 		return xx
	# 	return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def _entity_extraction_layer(self, features, scope, name="", share_trainables=True):
		layer_type = 'EntityExtraction'
		def layer_fn():
			# [B,Height,W,D]
			x = self._cnn_layer(input=features, scope=scope, name=name, share_trainables=share_trainables)
			# [B,Height,W,D+3]
			coordinates = self.get_coordinates(x)
			x = tf.concat([x, coordinates], axis=-1)
			# [B,N,D+3] N=Height*w
			_, h, w, ext_channels = x.shape.as_list()
			entities = tf.reshape(x, [-1, h * w, ext_channels])
			return entities
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def _relation_extraction_layer(self, entities, comparator_fn, edge_size_per_object_pair, n_object_pairs, scope, name="", share_trainables=True):
		layer_type = 'RelationExtraction'
		def layer_fn():
			key_size = EDGE_SIZE_PER_OBJECT_PAIR #entities.shape.as_list()[-1] # channels+3
			# What exactly are keys, queries, and values in attention mechanisms? https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms
			queries = self.__query_layer(
				n_query=2, 
				entities=entities, 
				n_object_pairs=n_object_pairs, 
				key_size=key_size, 
				scope=scope, share_trainables=share_trainables
			)
			# (batch_size, heads, n_query, key_size)
			keys = self.__key_layer(
				entities=entities, 
				n_object_pairs=n_object_pairs, 
				key_size=key_size, 
				scope=scope, share_trainables=share_trainables
			)
			# (batch_size, heads, height*width, key_size)
			values = tf.tile(tf.expand_dims(entities, 1), [1, n_object_pairs, 1, 1])
			# (batch_size, heads, height*width, channels+3)

			# Compute a pair of features using attention weights # objects = tf.keras.layers.Attention()([queries,values,keys])
			scores = tf.matmul(queries, keys, transpose_b=True)
			# (batch_size, heads, n_query, height*width)
			attention_weights = tf.nn.softmax(scores)
			# (batch_size, heads, n_query, height*width)
			objects = tf.matmul(attention_weights, values)
			# (batch_size, heads, n_query, (channels+3))

			# Spatial embedding
			objects_embedding = tf.keras.layers.Dense(
				units=edge_size_per_object_pair, 
				use_bias=False,
				activation=tf.nn.relu, # non-linear mapping
				kernel_initializer=self.kernel_initializer,
			)(objects)
			# (batch_size, heads, n_query, relations)

			# Comparator
			object1_embedding, object2_embedding = tf.unstack(objects_embedding, axis=2)
			differences = comparator_fn(object1_embedding, object2_embedding)

			# Positions
			object_positions = tf.slice(objects, [0, 0, 0, objects.shape.as_list()[-1]-3], [-1, -1, -1, -1]) # task,x,y coordinates are the last 3
			assert object_positions.shape.as_list()[-1] == 3, f"Error: wrong positions, they should have length 3 but now it is {object_positions.shape.as_list()[-1]}"
			# (batch_size, heads, n_query, 3)
			pos_obj1, pos_obj2 = tf.unstack(object_positions, axis=2)
			# (batch_size, heads, 3)

			# Collect differences and concatenate positions (objects)
			triples = tf.concat([differences, pos_obj1, pos_obj2], -1)
			# (batch_size, heads, differences+6)
			return triples, attention_weights
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def __key_layer(self, entities, n_object_pairs, key_size, scope, name="", share_trainables=True):
		layer_type = 'Key'
		def layer_fn():
			key = tf.keras.layers.Dense(
				units=key_size, 
				activation=tf.nn.relu, # non-linear mapping
				use_bias=False,
				kernel_initializer=self.kernel_initializer,
			)(entities)
			# (batch_size, h*w, key_size)
			key = tf.tile(tf.expand_dims(key, 1), [1, n_object_pairs, 1, 1])
			# (batch_size, n_object_pairs, height*width, key_size)
			return key
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def __query_layer(self, n_query, entities, n_object_pairs, key_size, scope, name="", share_trainables=True):
		layer_type = 'Query'
		def layer_fn():
			# Queries
			flatten_entities = tf.layers.flatten(entities)
			
			query = tf.keras.layers.Dense(
				units=n_object_pairs*n_query*key_size, 
				activation=tf.nn.relu, # non-linear mapping
				use_bias=False,
				kernel_initializer=self.kernel_initializer,
			)(flatten_entities)
			# (batch_size, n_object_pairs*n_query*key_size)
			query = tf.reshape(query, [-1, n_object_pairs, n_query, key_size])
			# (batch_size, n_object_pairs, n_query, key_size)
			return query
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	@staticmethod
	def get_coordinates(input_tensor, normalized=True, task_id=0):
		"""
		The output of cnn is tagged with two extra channels indicating the spatial position(x and y) of each cell

		:param input_tensor: (TensorFlow Tensor)  [B,Height,W,D]
		:return: (TensorFlow Tensor) [B,Height,W,3]
		"""
		batch_size = tf.shape(input_tensor)[0]
		_, height, width, _ = input_tensor.shape.as_list()
		if normalized:
			coor = [[[task_id, h / height, w / width] for w in range(width)] for h in range(height)]
		else:
			coor = [[[task_id, h, w] for w in range(width)] for h in range(height)]
		coor = tf.expand_dims(tf.constant(coor, dtype=input_tensor.dtype), axis=0)
		coor = tf.convert_to_tensor(coor)
		# [1,Height,W,3] --> [B,Height,W,3]
		coor = tf.tile(coor, [batch_size, 1, 1, 1])
		return coor

	def _concat_layer(self, input, concat, scope, name="", share_trainables=True):
		channels = input.shape.as_list()[-1] - 3

		layer_type = 'Concat'
		def layer_fn():
			new_concat = [
				tf.keras.layers.Dense(
					name=f'ConcatDenseEmbedding{i}', 
					units=channels, 
					activation=tf.nn.relu, 
					kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2))
				)(c)
				for i,c in enumerate(concat)
			]
			new_concat = list(map(tf.layers.flatten, new_concat))
			new_concat = tf.stack(new_concat)
			# (y, batch_size, channels)
			new_concat = tf.transpose(new_concat, [1,0,2])
			# (batch_size, y, channels)
			new_concat = tf.expand_dims(new_concat,1)
			# (batch_size, 1, y, channels)
			new_concat = tf.concat([new_concat, self.get_coordinates(new_concat, task_id=1)], axis=-1)
			# (batch_size, 1, y, channels+3)
			new_concat = tf.squeeze(new_concat,1)
			# (batch_size, y, channels+3)
			new_input = tf.concat([new_concat,input],1) # new_input = tf.concat([new_concat,input],-1) # do not do tf.concat([input, new_concat],-1) # (batch_size, width*height, channels + 2 + y)
			# (batch_size, y + width*height, channels+3)
			return new_input
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def _relational_layer(self, state, concat, scope, name="", share_trainables=True):
		layer_type = 'Relational'
		def layer_fn():
			entities = self._entity_extraction_layer(features=state, scope=scope, name='EE_1', share_trainables=share_trainables)
			# Concatenate extra features
			if len(concat) > 0:
				entities = self._concat_layer(input=entities, concat=concat, scope=scope, name="C_1", share_trainables=share_trainables)
			print( "	[{}]Entity Extraction layer {} output shape: {}".format(self.id, name, entities.get_shape()) )
			relations, attention_weights = self._relation_extraction_layer(entities, comparator_fn=tf.subtract, edge_size_per_object_pair=EDGE_SIZE_PER_OBJECT_PAIR, n_object_pairs=OBJECT_PAIRS, scope=scope, name="RE_1", share_trainables=share_trainables)
			print( "	[{}]Relation Extraction layer {} output shape: {}".format(self.id, name, relations.get_shape()) )
			output = relations
			relations_set = {
				'entity': {
					'relations': relations,
					'attention_weights': attention_weights
				},
			}
			return output, relations_set
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)	

	def _state_embedding_layer(self, state_batch, concat_batch):
		# Extract features
		relations_batch,self.relations_sets = zip(*[
			self._relational_layer(
				state = substate_batch/self.state_scaler, 
				concat = concat_batch,
				name = f'RE_{i}', 
				scope = self.parent_scope_name, 
			)
			for i,substate_batch in enumerate(state_batch)	
		])
		relations_batch = list(map(tf.layers.flatten, relations_batch))
		relations_batch = tf.stack(relations_batch)
		relations_batch = tf.transpose(relations_batch, [1,0,2])
		embedded_input = tf.layers.flatten(relations_batch)
		# print( "	[{}]State Relational layer output shape: {}".format(self.id, embedded_input.get_shape()) )
		# embedded_input = tf.keras.layers.Dense(name='DenseEmbedding',  units=256, activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))(embedded_input)
		# [Training state]
		if flags.use_learnt_environment_model_as_observation:
			embedded_input = self._weights_layer(input=embedded_input, weights=self.training_state, scope=self.parent_scope_name)
			# print( "	[{}]Weights layer output shape: {}".format(self.id, embedded_input.get_shape()) )
		return embedded_input
