# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.compat.v1 as tf
from agent.network.actor_critic.explicitly_relational_network import ExplicitlyRelational_Network
import options
flags = options.get()

HEADS = 2
RELATIONS_PER_HEAD = 16

# Shanahan, Murray, et al. "An explicitly relational neural network architecture." arXiv preprint arXiv:1905.10307 (2019).
class ExplicitlyArgumentative_Network(ExplicitlyRelational_Network):
	
	def _relational_layer(self, input, scope, name="", share_trainables=True):
		layer_type = 'RelationalNet'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			entities = self.entity_extraction_layer(input, scope, name='EntityExtraction', share_trainables=share_trainables)
			print( "	Entity Extraction layer output shape: {}".format(entities.get_shape()) )
			relations = self.relation_extraction_layer(entities, n_relations=RELATIONS_PER_HEAD, n_heads=HEADS)
			print( "	Relation Extraction layer shape: {}".format(relations.get_shape()) )
			argumentation_links = self.relation_extraction_layer(relations, n_relations=RELATIONS_PER_HEAD, n_heads=HEADS)
			print( "	Arguments Links Extraction layer shape: {}".format(argumentation_links.get_shape()) )
			relations = tf.concat([relations,argumentation_links], -1)
			relations = tf.layers.flatten(relations)
			# update keys
			self._update_keys(variable_scope.name, share_trainables)
			# return result
			return relations
	