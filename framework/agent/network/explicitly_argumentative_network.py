# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.compat.v1 as tf
import utils.tensorflow_utils as tf_utils
from agent.network.explicitly_relational_network import ExplicitlyRelational_Network
import options
flags = options.get()

OBJECT_PAIRS = 16
EDGE_SIZE_PER_OBJECT_PAIR = 4

class ExplicitlyArgumentative_Network(ExplicitlyRelational_Network):
	
	def argument_link_extraction_layer(self, relations, n_links, n_object_pairs, scope, name="", share_trainables=True):
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

		layer_type = 'ArgLinkExtraction'
		def layer_fn():
			arguments = tf.expand_dims(relations,-1)
			arguments = tf.concat([arguments, self.get_coordinates(arguments, task_id=0.5)], axis=-1)
			_, h, w, ext_channels = arguments.shape.as_list()
			arguments = tf.reshape(arguments, [-1, h * w, ext_channels])
			return self._relation_extraction_layer(arguments, comparator_fn=argument_comparator_fn, edge_size_per_object_pair=n_links, n_object_pairs=n_object_pairs, scope=scope, name="RE_1", share_trainables=share_trainables)
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def _relational_layer(self, state, concat, scope, name="", share_trainables=True):
		layer_type = 'Relational'
		def layer_fn():
			entities = [
				self._entity_extraction_layer(
					features=substate/self.state_scaler, 
					scope=self.parent_scope_name, 
					name=f'EE_{i}', 
					share_trainables=share_trainables
				)
				for i,substate in enumerate(state)
			]
			entities = tf.concat(entities,1)
			# Concatenate extra features
			if len(concat) > 0:
				entities = self._concat_layer(input=entities, concat=concat, scope=scope, name="C_1", share_trainables=share_trainables)
			print( "	[{}]Entity Extraction layer {} output shape: {}".format(self.id, name, entities.get_shape()) )
			entity_relations, entity_attention_weights = self._relation_extraction_layer(entities=entities, comparator_fn=tf.subtract, edge_size_per_object_pair=EDGE_SIZE_PER_OBJECT_PAIR, n_object_pairs=OBJECT_PAIRS, scope=scope, name="RE_1", share_trainables=share_trainables)
			print( "	[{}]Relation Extraction layer {} output shape: {}".format(self.id, name, entity_relations.get_shape()) )
			argument_links, relation_attention_weights = self.argument_link_extraction_layer(relations=entity_relations, n_links=EDGE_SIZE_PER_OBJECT_PAIR, n_object_pairs=OBJECT_PAIRS, scope=scope, name="ALE_1", share_trainables=share_trainables)
			print( "	[{}]Argument Link Extraction layer {} output shape: {}".format(self.id, name, argument_links.get_shape()) )
			output = tf.concat([entity_relations,argument_links], 1)
			relations_set = {
				'entity': {
					'relations': entity_relations,
					'attention_weights': entity_attention_weights
				},
				'relation': {
					'relations': argument_links,
					'attention_weights': relation_attention_weights
				},
			}
			return output, relations_set
		return self._scopefy(output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)
	