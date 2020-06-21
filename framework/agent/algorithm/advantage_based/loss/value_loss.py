# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
import utils.tensorflow_utils as tf_utils

import options
flags = options.get()

class ValueLoss(object):
	def __init__(self, global_step, loss, target, prediction, old_prediction=None):
		self.global_step = global_step
		self.loss = loss.lower()
		self.prediction = prediction
		# Stop gradients
		if old_prediction != None:
			self.old_prediction = tf.stop_gradient(old_prediction)
		self.target = tf.stop_gradient(target)
		# Get reduce function
		self.reduce_function = eval('tf.reduce_{}'.format(flags.loss_type))
		
	def get(self):
		return eval('self.{}'.format(self.loss))()
			
	def vanilla(self):
		return self.reduce_function(tf.keras.losses.MSE(self.target, self.prediction))
				
	# Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
	def pvo(self):
		# clip
		clip_range = tf_utils.get_annealable_variable(
			function_name=flags.clip_annealing_function, 
			initial_value=flags.clip, 
			global_step=self.global_step, 
			decay_steps=flags.clip_decay_steps, 
			decay_rate=flags.clip_decay_rate
		) if flags.clip_decay else flags.clip
		clip_range = tf.cast(clip_range, self.prediction.dtype)
		# clipped prediction
		prediction_clipped = self.old_prediction + tf.clip_by_value(self.prediction-self.old_prediction, -clip_range, clip_range)
		max_delta = tf.maximum(tf.abs(self.target-self.prediction),tf.abs(self.target-prediction_clipped))
		return self.reduce_function(tf.reduce_mean(tf.square(max_delta), -1))
