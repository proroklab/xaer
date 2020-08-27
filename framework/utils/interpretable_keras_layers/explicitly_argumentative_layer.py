# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from utils.interpretable_keras_layers.explicitly_relational_layer import ExplicitlyRelationalLayer, OR,NOR,AND,NAND,XOR,XNOR

class ExplicitlyArgumentativeLayer(tf.keras.layers.Layer):

	@staticmethod
	def argument_comparator_fn(l,r):
		similarity_relation = tf.subtract(l,r)
		# priority_relation = tf.cast(tf.greater(l,r), tf.float32)
		# inverse_priority_relation = tf.cast(tf.less(l,r), tf.float32)
		# return similarity_relation # V1
		# return tf.concat([similarity_relation, priority_relation], -1) # V2
		# return tf.concat([tf.nn.relu(similarity_relation), priority_relation], -1) # V3
		# return tf.concat([tf.minimum(0,similarity_relation), priority_relation], -1) # V4
		return tf.nn.relu(similarity_relation) # V5, same of: tf.multiply(similarity_relation, priority_relation)
		# return tf.concat([tf.minimum(0,similarity_relation), tf.nn.relu(similarity_relation)], -1) # V6
		# return priority_relation # V7
		# return tf.minimum(0.,similarity_relation) # V8, same of: tf.multiply(similarity_relation, inverse_priority_relation)

	def __init__(self, 
			name=None, 
			return_explicit_model=False,
			object_pairs=16, 
			edge_size_per_object_pair=4, 
			operators_set=[OR,NOR,AND,NAND,XOR,XNOR], 
			argument_links=8, 
			debug=False,
			**kwargs
		):
		super(ExplicitlyArgumentativeLayer, self).__init__(name=name, **kwargs)
		self.debug = debug
		self.return_explicit_model = return_explicit_model
		self.object_pairs = object_pairs
		self.argument_links = argument_links
		self.edge_size_per_object_pair = edge_size_per_object_pair
		self.operators_set = operators_set

	def compute_output_shape(self, input_shape):
		return [input_shape[0],self.object_pairs+self.argument_links,self.edge_size_per_object_pair]

	def build(self, input_shape):  # Create the state of the layer (weights)
		self.rel1 = ExplicitlyRelationalLayer(
			name='Concepts_ERL', 
			return_explicit_model=True,
			object_pairs=self.object_pairs, 
			edge_size_per_object_pair=self.edge_size_per_object_pair, 
			operators_set=self.operators_set
		)
		self.rel2 = ExplicitlyRelationalLayer(
			name='Arguments_ERL',
			return_explicit_model=True, 
			object_pairs=self.argument_links, 
			edge_size_per_object_pair=self.edge_size_per_object_pair, 
			operators_set=[self.argument_comparator_fn]
		)

	def call(self, inputs):
		augmented_relations, explicit_model = self._relation_extraction_block(inputs)
		if self.debug:
			print( "	Argumentative layer {} output shape: {}".format(self.name, augmented_relations.get_shape()) )
		if self.return_explicit_model:
			return augmented_relations, explicit_model
		return augmented_relations
	
	def _relation_extraction_block(self, features):
		relations,explicit_conceptual_model = self.rel1(features)
		triples = explicit_conceptual_model['conceptual']['triple_list']
		_,h,r,_ = triples.shape.as_list()
		triples = tf.reshape(triples, [-1,h*r,3])
		if self.debug:
			print( "	Triples shape: {}".format(triples.get_shape()) )
		arguments,explicit_argumentative_model = self.rel2(triples)
		output = tf.concat([relations,arguments], 1) # residual block -> this might give a logical interpretation of the purposes of a residual block
		explicit_model = {
			'conceptual': explicit_conceptual_model['conceptual'],
			'argumentative': explicit_argumentative_model['conceptual'],
		}
		return output, explicit_model
	