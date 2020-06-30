# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.compat.v1 as tf
import utils.tensorflow_utils as tf_utils
from agent.network.openai_small_network import OpenAISmall_Network
import options
flags = options.get()

def NOT(x):
	return 1. - x

def AND(x,y):
	return tf.minimum(x,y) # x*y

def OR(x,y):
	return tf.maximum(x,y) # x + y - AND(x,y)

def NAND(x,y):
	return NOT(AND(x,y)) # 1. - x*y

def NOR(x,y):
	return NOT(OR(x,y)) # 1. - x + y - AND(x,y)

def XOR(x,y):
	return AND(OR(x,y), OR(NOT(x), NOT(y)))

def XNOR(x,y):
	return OR(AND(x,y), AND(NOT(x), NOT(y))) # NOT(XOR(x,y))

# Shanahan, Murray, et al. "An explicitly relational neural network architecture." arXiv preprint arXiv:1905.10307 (2019).
class ExplicitlyRelational_Network(OpenAISmall_Network):
	produce_explicit_relations = True
	kernel_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1)

	def __init__(self, id, policy_heads, scope_dict, training=True, value_count=1, state_scaler=1):
		super().__init__(id, policy_heads, scope_dict, training, value_count, state_scaler)
		self.object_pairs = 16
		self.edge_size_per_object_pair = 2
		self.relational_layer_operators_set = [OR,NOR,AND,NAND,XOR,XNOR]

	def _entity_extraction_layer(self, features, scope="", name="", share_trainables=True):
		layer_type = 'EntityExtraction'
		def layer_fn():
			# [B,Height,W,D]
			x = self._cnn_layer(input=features, share_trainables=share_trainables)
			# [B,Height,W,D+3]
			coordinates = self.get_coordinates(x)
			x = tf.concat([x, coordinates], axis=-1)
			# [B,N,D+3] N=Height*w
			_, h, w, ext_channels = x.shape.as_list()
			entities = tf.reshape(x, [-1, h * w, ext_channels])
			return entities
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def _relation_extraction_layer(self, entities, operators_set, edge_size_per_object_pair, n_object_pairs, scope="", name="", share_trainables=True):
		layer_type = 'RelationExtraction'
		def layer_fn():
			# What exactly are keys, queries, and values in attention mechanisms? https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms
			queries = self.__query_layer(
				n_query=2, # binary relations
				entities=entities, 
				n_object_pairs=n_object_pairs, 
				key_size=entities.shape.as_list()[-1], # channels+3, 
				share_trainables=share_trainables,
			)
			# (batch_size, heads, n_query, channels+3)
			keys = values = tf.tile(tf.expand_dims(entities, 1), [1, n_object_pairs, 1, 1])
			# (batch_size, heads, height*width, channels+3)

			# Compute a pair of features using self-attention weights # objects = tf.keras.layers.Attention()([queries,values,keys])
			scores = tf.matmul(queries, keys, transpose_b=True)
			# (batch_size, heads, n_query, height*width)
			attention_weights = tf.nn.softmax(scores)
			# (batch_size, heads, n_query, height*width)
			objects = tf.matmul(attention_weights, values)
			# (batch_size, heads, n_query, (channels+3))

			# Spatial embedding
			objects_embedding = tf.keras.layers.Dense( # negation operator is embedded in here
				name='Dense1',
				units=edge_size_per_object_pair, 
				# use_bias=True,
				# activation=tf.nn.relu, # non-linear mapping
				kernel_initializer=self.kernel_initializer,
			)(objects)
			# (batch_size, heads, n_query, relations)

			# Normalize in [0,1]
			objects_embedding = (1. + tf.keras.layers.LayerNormalization(name='LayerNorm_1')(objects_embedding))/2.
			objects_embedding = tf.clip_by_value(objects_embedding, 0., 1.)
			# (batch_size, heads, n_query, relations)

			object1_embedding, object2_embedding = tf.unstack(objects_embedding, axis=2)
			# (batch_size, heads, relations)
			if len(operators_set) > 1:
				print( "	[{}]Relation Extraction layer {} operators: {}".format(self.id, name, operators_set) )
				objects_embedding_shape = objects_embedding.shape.as_list()
				reshaped_operator_logits = tf.reshape(objects_embedding, [-1, objects_embedding_shape[1], np.prod(objects_embedding_shape[2:])])
				# (batch_size, heads, n_query*relations)
				operator_logits = tf.keras.layers.Dense(
					name='Dense2',
					units=edge_size_per_object_pair*len(operators_set), 
					# use_bias=True,
					# activation=tf.nn.relu, # non-linear mapping
					kernel_initializer=tf_utils.orthogonal_initializer()
				)(reshaped_operator_logits)
				# (batch_size, heads, relations*ops)
				operator_logits = tf.reshape(operator_logits, [-1, operator_logits.shape.as_list()[1], edge_size_per_object_pair, len(operators_set)])
				# (batch_size, heads, relations, ops)
				operator_logits = tf.keras.layers.LayerNormalization(name='LayerNorm_2')(operator_logits)
				operators_mask = tf.nn.softmax(operator_logits*1e2, -1)
				# (batch_size, heads, relations, ops)
				operators_result = tf.stack([op(object1_embedding, object2_embedding) for i,op in enumerate(operators_set)], -1)
				# (batch_size, heads, relations, ops)
				result = tf.reduce_sum(tf.multiply(operators_result,operators_mask), -1)
				# (batch_size, heads, relations)
			else:
				result = operators_set[0](object1_embedding, object2_embedding)

			# Positions
			object_positions = tf.slice(objects, [0, 0, 0, objects.shape.as_list()[-1]-3], [-1, -1, -1, -1]) # task,x,y coordinates are the last 3
			assert object_positions.shape.as_list()[-1] == 3, f"Error: wrong positions, they should have length 3 but now it is {object_positions.shape.as_list()[-1]}"
			# (batch_size, heads, n_query, 3)
			pos_obj1, pos_obj2 = tf.unstack(object_positions, axis=2)
			# (batch_size, heads, 3)

			# Collect differences and concatenate positions (objects)
			triples = tf.concat([result, pos_obj1, pos_obj2], -1)
			# (batch_size, heads, differences+6)
			return triples, attention_weights
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def __key_layer(self, entities, n_object_pairs, key_size, scope="", name="", share_trainables=True):
		layer_type = 'Key'
		def layer_fn():
			key = tf.keras.layers.Dense(
				units=key_size, 
				# activation=tf.nn.relu, # non-linear mapping
				use_bias=False,
				kernel_initializer=self.kernel_initializer,
			)(entities)
			# (batch_size, h*w, key_size)
			key = tf.tile(tf.expand_dims(key, 1), [1, n_object_pairs, 1, 1])
			# (batch_size, n_object_pairs, height*width, key_size)
			return key
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def __query_layer(self, n_query, entities, n_object_pairs, key_size, scope="", name="", share_trainables=True):
		layer_type = 'Query'
		def layer_fn():
			# Queries
			flatten_entities = tf.keras.layers.Flatten()(entities)
			
			query = tf.keras.layers.Dense(
				units=n_object_pairs*n_query*key_size, 
				# activation=tf.nn.relu, # non-linear mapping
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

	def _concat_layer(self, input, concat, scope="", name="", share_trainables=True):
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
			new_concat = list(map(tf.keras.layers.Flatten(), new_concat))
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

	def _relational_layer(self, state, concat, scope="", name="", share_trainables=True):
		layer_type = 'Relational'
		def layer_fn():
			entities = [
				self._entity_extraction_layer(
					features=substate/self.state_scaler, 
					name=f'EE_{i}', 
					share_trainables=share_trainables
				)
				for i,substate in enumerate(state)
			]
			entities = tf.concat(entities,1)
			# Concatenate extra features
			if len(concat) > 0:
				entities = self._concat_layer(
					input=entities, 
					concat=concat, 
					share_trainables=share_trainables
				)
			print( "	[{}]Entity Extraction layer {} output shape: {}".format(self.id, name, entities.get_shape()) )
			relations, attention_weights = self._relation_extraction_layer(
				entities, 
				operators_set=self.relational_layer_operators_set, 
				edge_size_per_object_pair=self.edge_size_per_object_pair, 
				n_object_pairs=self.object_pairs, 
				share_trainables=share_trainables
			)
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

	def _state_embedding_layer(self, state_batch, concat_batch, environment_model, scope="", name="", share_trainables=True):
		layer_type = 'StateEmbedding'
		def layer_fn():
			# Extract features
			relations_batch,self.relations_sets = self._relational_layer(
				state=state_batch, 
				concat=concat_batch,
				share_trainables=share_trainables,
			)
			embedded_input = tf.keras.layers.Flatten()(relations_batch)
			if flags.use_learnt_environment_model_as_observation:
				embedded_input = self._weights_layer(
					input=embedded_input, 
					weights=environment_model, 
					share_trainables=share_trainables,
				)
			return embedded_input
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)
