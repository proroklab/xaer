# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
import tensorflow.keras.layers as ktf
import numpy as np
import utils.tensorflow_utils as tf_utils
from agent.network.network import Network

class IntrinsicReward_Network(Network):
	def __init__(self, id, batch_dict, scope_dict, training=True):
		super().__init__(id, training)
		self.scope_name = scope_dict['self']
		# Shape network
		self.state_batch = batch_dict['new_state']
		self.state_mean_batch = batch_dict['state_mean']
		self.state_std_batch = batch_dict['state_std']
		
	def build(self):
		# Use state_batch instead of new_state_batch, to save memory
		normalized_state_batch = (self.state_batch[0]-self.state_mean_batch[0][-1])/self.state_std_batch[0][-1]
		# normalized_state_batch = normalized_state_batch[:, :, :, -1:]
		normalized_state_batch = tf.clip_by_value(normalized_state_batch, -5.0, 5.0)
		# Build layer
		target, prediction, training_state = self._intrinsic_reward_layer(normalized_state_batch, scope=self.scope_name)
		noisy_target = tf.stop_gradient(target)
		#=======================================================================
		# # Get feature variance
		# feature_variance = tf.reduce_mean(tf.nn.moments(target, axes=[0])[1])
		# # Get maximum feature
		# max_feature = tf.reduce_max(tf.abs(target))
		#=======================================================================
		# Get intrinsic reward
		intrinsic_reward = tf.reduce_mean(tf.squared_difference(noisy_target,prediction), axis=-1)
		# Get loss
		loss = tf.reduce_mean(tf.nn.dropout(intrinsic_reward, 0.5))
		# loss = tf.reduce_mean(intrinsic_reward)
		# Return results
		intrinsic_reward = tf.reshape(intrinsic_reward, [-1])
		return intrinsic_reward, loss, training_state
	
	def _intrinsic_reward_layer(self, input, scope, name="", share_trainables=True):
		layer_type = 'RandomNetworkDistillation'
		with tf.variable_scope("{}/{}{}".format(scope,layer_type,name), reuse=tf.AUTO_REUSE) as variable_scope:
			print( "	[{}]Building or reusing scope: {}".format(self.id, variable_scope.name) )
			# Here we use leaky_relu instead of relu as activation function
			# Target network
			target = ktf.Conv2D(name='RND_Target_Conv1', filters=32, kernel_size=8, strides=4, padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2))).apply(inputs=input)
			target = ktf.Conv2D(name='RND_Target_Conv2', filters=64, kernel_size=4, strides=2, padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2))).apply(inputs=target)
			target = ktf.Conv2D(name='RND_Target_Conv3', filters=64, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2))).apply(inputs=target)
			target = ktf.Flatten().apply(inputs=target)
			target = ktf.Dense(name='RND_Target_Dense1', units=512, activation=None, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2))).apply(inputs=target)
			# Predictor network
			prediction = ktf.Conv2D(name='RND_Prediction_Conv1', filters=32, kernel_size=8, strides=4, padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2))).apply(inputs=input)
			prediction = ktf.Conv2D(name='RND_Prediction_Conv2', filters=64, kernel_size=4, strides=2, padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2))).apply(inputs=prediction)
			prediction = ktf.Conv2D(name='RND_Prediction_Conv3', filters=64, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2))).apply(inputs=prediction)
			prediction = ktf.Flatten().apply(inputs=prediction)
			prediction = ktf.Dense(name='RND_Prediction_Dense1', units=512, activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2))).apply(inputs=prediction)
			prediction = ktf.Dense(name='RND_Prediction_Dense2', units=512, activation=tf.nn.relu, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2))).apply(inputs=prediction)
			last_prediction_layer = ktf.Dense(name='RND_Prediction_Dense3', units=512, activation=None, kernel_initializer=tf_utils.orthogonal_initializer(np.sqrt(2)))
			prediction = last_prediction_layer.apply(inputs=prediction)
			prediction_weights = {
				'kernel': last_prediction_layer.kernel, 
				'bias': last_prediction_layer.bias
			}
			# update keys
			self._update_keys(variable_scope.name, share_trainables)
			# return result
			return target, prediction, prediction_weights
