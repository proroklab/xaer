# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.compat.v1 as tf

import utils.tensorflow_utils as tf_utils
from agent.network.openai_small_network import OpenAISmall_Network
from utils.interpretable_keras_layers.explicitly_argumentative_layer import ExplicitlyArgumentativeLayer, ExplicitlyRelationalLayer, OR,NOR,AND,NAND,XOR,XNOR
import options
flags = options.get()

# Shanahan, Murray, et al. "An explicitly relational neural network architecture." arXiv preprint arXiv:1905.10307 (2019).
class ExplicitlyRelational_Network(OpenAISmall_Network):
	produce_explicit_relations = True
	kernel_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1)

	def __init__(self, id, policy_heads, scope_dict, training=True, value_count=1, state_scaler=1):
		super().__init__(id, policy_heads, scope_dict, training, value_count, state_scaler)
		self.object_pairs = 16
		self.edge_size_per_object_pair = 4
		self.relational_layer_operators_set = [OR,NOR,AND,NAND,XOR,XNOR]

	def _entity_extraction_layer(self, features, scope="", name="", share_trainables=True):
		# [B,Height,W,D]
		x = self._cnn_layer(input=features, share_trainables=share_trainables, name=name)
		# [B,N,D] N=Height*w
		_, h, w, ext_channels = x.shape.as_list()
		entities = tf.reshape(x, [-1, h * w, ext_channels])
		return entities

	def _concat_layer(self, input, concat, scope="", name="", share_trainables=True):
		channels = input.shape.as_list()[-1]
		layer_type = 'Concat'
		def layer_fn():
			concat_layer = [
				tf.keras.Sequential([
					tf.keras.layers.Dense(
						name=f'ConcatDenseEmbedding{i}', 
						units=channels, 
						activation=tf.nn.relu, 
						kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2))
					),
					tf.keras.layers.Flatten(),
				])
				for i,c in enumerate(concat)
			]
			def exec_fn(i, c):
				c = [l(x) for l,x in zip(concat_layer, c)]
				c = tf.stack(c, 1)
				# (batch_size, y, channels)
				i = tf.concat([c,i],1) # new_input = tf.concat([new_concat,input],-1) # do not do tf.concat([input, new_concat],-1) # (batch_size, width*height, channels + 2 + y)
				# (batch_size, y + width*height, channels)
				return i
			return exec_fn
		return self._scopefy(inputs=(input, concat), output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def _relational_layer(self, state, concat, scope="", name="", share_trainables=True):
		layer_type = 'Relational'
		def layer_fn():
			relational_layer = ExplicitlyRelationalLayer(
				name='RelNet1',
				object_pairs=self.object_pairs, 
				edge_size_per_object_pair=self.edge_size_per_object_pair, 
				operators_set=self.relational_layer_operators_set, 
			)
			def exec_fn(s, c):
				entities = [
					self._entity_extraction_layer(
						features=substate/self.state_scaler, 
						name=f'EE_{i}', 
						share_trainables=share_trainables
					)
					for i,substate in enumerate(s)
				]
				entities = tf.concat(entities,1)
				# Concatenate extra features
				if len(c) > 0:
					entities = self._concat_layer(
						input=entities, 
						concat=c, 
						share_trainables=share_trainables
					)
				print( "	[{}]Entity Extraction layer {} output shape: {}".format(self.id, name, entities.get_shape()) )
				relations = relational_layer(entities)
				print( "	[{}]Relation Extraction layer {} output shape: {}".format(self.id, name, relations.get_shape()) )
				return relations,{}
			return exec_fn
		return self._scopefy(inputs=(state, concat), output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)

	def _state_embedding_layer(self, state_batch, concat_batch, environment_model, scope="", name="", share_trainables=True):
		layer_type = 'StateEmbedding'
		def layer_fn():
			def exec_fn(s, c):
				# Extract features
				relations_batch,self.relations_sets = self._relational_layer(
					state=s, 
					concat=c,
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
			return exec_fn
		return self._scopefy(inputs=(state_batch, concat_batch), output_fn=layer_fn, layer_type=layer_type, scope=scope, name=name, share_trainables=share_trainables)
