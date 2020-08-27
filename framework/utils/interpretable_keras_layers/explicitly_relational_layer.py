# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

def NOT(x):
	return 1. - x

def AND(x,y):
	return tf.minimum(x,y) # x*y

def OR(x,y):
	return tf.maximum(x,y) # x + y - AND(x,y)

def NAND(x,y):
	return NOT(AND(x,y))

def NOR(x,y):
	return NOT(OR(x,y))

def XOR(x,y):
	return AND(OR(x,y), OR(NOT(x), NOT(y)))

def XNOR(x,y):
	return OR(AND(x,y), AND(NOT(x), NOT(y))) # NOT(XOR(x,y))

# Shanahan, Murray, et al. "An explicitly relational neural network architecture." arXiv preprint arXiv:1905.10307 (2019).
class ExplicitlyRelationalLayer(tf.keras.layers.Layer):
	relation_arity = 2

	def __init__(self, 
			name=None, 
			return_explicit_model=False, 
			object_pairs=16, 
			edge_size_per_object_pair=4, 
			operators_set=[OR,NOR,AND,NAND,XOR,XNOR], 
			debug=False, 
			**kwargs
		):
		super(ExplicitlyRelationalLayer, self).__init__(name=name, **kwargs)
		self.debug = debug
		self.return_explicit_model = return_explicit_model
		self.object_pairs = object_pairs
		self.edge_size_per_object_pair = edge_size_per_object_pair
		self.operators_set = operators_set
		self.kernel_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1)

	def compute_output_shape(self, input_shape):
		return [input_shape[0],self.object_pairs,self.edge_size_per_object_pair]

	def build(self, input_shape):  # Create the state of the layer (weights)
		self.dense1 = tf.keras.layers.Dense( # negation operator is embedded in here
			name='Dense1',
			units=self.edge_size_per_object_pair, 
			# use_bias=True,
			# activation=tf.nn.relu, # non-linear mapping
			kernel_initializer=self.kernel_initializer,
		)
		self.dense2 = tf.keras.layers.Dense(
			name='Dense2',
			units=self.edge_size_per_object_pair*len(self.operators_set), 
			# use_bias=True,
			# activation=tf.nn.relu, # non-linear mapping
			kernel_initializer=self.kernel_initializer
		)
		self.dense3 = tf.keras.layers.Dense(
			name='Dense3',
			units=self.object_pairs*self.relation_arity, 
			# activation=tf.nn.relu, # non-linear mapping
			use_bias=False,
			kernel_initializer=self.kernel_initializer,
		)
		self.layer_norm1 = tf.keras.layers.LayerNormalization(name='LayerNorm_1')
		self.layer_norm2 = tf.keras.layers.LayerNormalization(name='LayerNorm_2')

	def call(self, inputs):
		relations, explicit_model = self._relation_extraction_block(inputs)
		if self.debug:
			print( "	Relation Extraction layer {} output shape: {}".format(self.name, relations.get_shape()) )
		if self.return_explicit_model:
			return relations, explicit_model
		return relations

	def _relation_extraction_block(self, features):
		# Format input
		while len(features.shape.as_list()) < 3:
			features = tf.expand_dims(features,-1)
		if len(features.shape.as_list()) > 3:
			shape = features.shape.as_list()
			features = tf.reshape(features, [-1, shape[1], np.prod(shape[2:])])

		# Get queries and keys
		# What exactly are keys, queries, and values in attention mechanisms? https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms
		features_shape = features.shape.as_list()
		key_size=features_shape[-1]
		flatten_features = tf.reshape(features, [-1,key_size,np.prod(features_shape[1:-1])])
		query = self.dense3(flatten_features)
		# (batch_size, n_object_pairs*n_query*key_size)
		queries = tf.reshape(query, [-1, self.object_pairs, self.relation_arity, key_size])
		# (batch_size, n_object_pairs, n_query, key_size)
		keys = values = tf.tile(tf.expand_dims(features, 1), [1, self.object_pairs, 1, 1])
		# (batch_size, n_object_pairs, height*width, channels)

		# Compute a pair of features using self-attention weights # objects = tf.keras.layers.Attention()([queries,values,keys])
		scores = tf.matmul(queries, keys, transpose_b=True)
		# (batch_size, heads, n_query, height*width)
		attention_weights = tf.nn.softmax(scores)
		# (batch_size, heads, n_query, height*width)
		objects = tf.matmul(attention_weights, values)
		# (batch_size, heads, n_query, channels)

		# Spatial embedding
		objects_embedding = self.dense1(objects)
		# (batch_size, heads, n_query, relations)

		# Normalize in [0,1]
		objects_embedding = (1. + self.layer_norm1(objects_embedding))/2. # roughly in [0,1]
		objects_embedding = tf.clip_by_value(objects_embedding, 0., 1.) # in [0,1]
		# (batch_size, heads, n_query, relations)

		object1_embedding, object2_embedding = tf.unstack(objects_embedding, axis=2)
		# (batch_size, heads, relations)
		if len(self.operators_set) > 1:
			if self.debug:
				print( "	Relation Extraction layer {} operators: {}".format(self.name, self.operators_set) )
			objects_embedding_shape = objects_embedding.shape.as_list()
			reshaped_operator_logits = tf.reshape(objects_embedding, [-1, objects_embedding_shape[1], np.prod(objects_embedding_shape[2:])])
			# (batch_size, heads, n_query*relations)
			operator_logits = self.dense2(reshaped_operator_logits)
			# (batch_size, heads, relations*ops)
			operator_logits = tf.reshape(operator_logits, [-1, operator_logits.shape.as_list()[1], self.edge_size_per_object_pair, len(self.operators_set)])
			operator_logits = self.layer_norm2(operator_logits) # roughly in [-1,1], this should guarantee the following gumbel softmax to properly approximate a discrete distribution
			# (batch_size, heads, relations, ops)
			operators_mask = tf.nn.softmax(operator_logits*1e2, -1) # gumbel softmax - https://casmls.github.io/general/2017/02/01/GumbelSoftmax.html # trick to not prevent/stop gradient flow from 'operators_mask' to 'operator_logits'
			# (batch_size, heads, relations, ops)
			operators_result = tf.stack([op(object1_embedding, object2_embedding) for i,op in enumerate(self.operators_set)], -1)
			# (batch_size, heads, relations, ops)
			result = tf.reduce_sum(tf.multiply(operators_result, operators_mask), -1) # trick to make gradient flow from 'result' to both 'operators_result' and 'operators_mask'
			# (batch_size, heads, relations)
			operators_id = tf.range(len(self.operators_set), dtype=tf.float32)
			# (ops)
			operator = tf.reduce_sum(tf.multiply(operators_id, operators_mask), -1) # trick to not prevent/stop gradient flow from 'operator' to 'operators_mask'
			# (batch_size, heads, relations)
		else:
			result = self.operators_set[0](object1_embedding, object2_embedding)
			# (batch_size, heads, relations)
			operator = tf.zeros_like(object1_embedding, tf.float32)
			# (batch_size, heads, relations)
		explicit_model = {
			'conceptual': {
				'values_list': objects_embedding,
				'ids_list': attention_weights,
				'operator_list': operator,
				'triple_list': tf.stack((object1_embedding, object2_embedding, operator), -1),
			}
		}
		return result, explicit_model
