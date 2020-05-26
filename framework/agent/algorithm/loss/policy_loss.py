# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
import utils.tensorflow_utils as tf_utils
from agent.network import is_continuous_control
from utils.distributions import Categorical, Normal

import options
flags = options.get()

class PolicyLoss(object):

	def __init__(self, global_step, type, policy_heads, actor_batch, old_policy_batch, old_action_batch, is_replayed_batch, old_action_mask_batch=None, beta=0):
		old_policy_distributions = []
		new_policy_distributions = []
		for h,policy_head in enumerate(policy_heads):
			if is_continuous_control(policy_head['depth']):
				# Old policy
				old_policy_batch_h = tf.transpose(old_policy_batch[h], [1, 0, 2])
				old_policy_distributions.append( Normal(old_policy_batch_h[0], old_policy_batch_h[1]) )
				# New policy
				new_policy_batch_h = tf.transpose(actor_batch[h], [1, 0, 2])
				new_policy_distributions.append( Normal(new_policy_batch_h[0], new_policy_batch_h[1]) )
			else: # discrete control
				old_policy_distributions.append( Categorical(old_policy_batch[h]) ) # Old policy
				new_policy_distributions.append( Categorical(actor_batch[h]) ) # New policy
		entropy = tf.squeeze(tf.transpose(
			tf.stack([d.entropy() for d in new_policy_distributions]),
			[1, 2, 0]
		), -1)
		print( "	Entropy shape: {}".format(entropy.get_shape()) )
		cross_entropy = tf.squeeze(tf.transpose(
			tf.stack([d.cross_entropy(old_action_batch[i]) for i,d in enumerate(new_policy_distributions)]),
			[1, 2, 0]
		), -1)
		print( "	Cross Entropy shape: {}".format(entropy.get_shape()) )
		old_cross_entropy = tf.squeeze(tf.transpose(
			tf.stack([d.cross_entropy(old_action_batch[i]) for i,d in enumerate(old_policy_distributions)]),
			[1, 2, 0]
		), -1)
		print( "	Old Cross Entropy shape: {}".format(old_cross_entropy.get_shape()) )
		if old_action_mask_batch is not None:
			old_action_mask_batch = tf.transpose(tf.stack(old_action_mask_batch), [1, 0])
			# stop gradient computation on masked elements and remove them from loss (zeroing)
			cross_entropy = tf.where(
				tf.equal(old_action_mask_batch,1),
				x=cross_entropy, # true branch
				y=tf.stop_gradient(old_action_mask_batch) # false branch
			)
			old_cross_entropy = tf.where(
				tf.equal(old_action_mask_batch,1),
				x=old_cross_entropy, # true branch
				y=tf.stop_gradient(old_action_mask_batch) # false branch
			)

		self.global_step = global_step
		self.type = type.replace(' ','_').lower()
		self.zero = tf.constant(0., dtype=cross_entropy.dtype)
		self.one = tf.constant(1., dtype=cross_entropy.dtype)
		self.is_replayed_batch = is_replayed_batch
		# Clip
		self.clip_range = tf_utils.get_annealable_variable(
			function_name=flags.clip_annealing_function, 
			initial_value=flags.clip, 
			global_step=self.global_step, 
			decay_steps=flags.clip_decay_steps, 
			decay_rate=flags.clip_decay_rate
		) if flags.clip_decay else flags.clip
		self.clip_range = tf.cast(self.clip_range, cross_entropy.dtype)
		# Entropy
		self.beta = beta
		self.entropy = tf.maximum(self.zero, entropy) if flags.only_non_negative_entropy else entropy
		# Cross Entropy
		self.cross_entropy = tf.maximum(self.zero, cross_entropy) if flags.only_non_negative_entropy else cross_entropy
		self.old_cross_entropy = tf.maximum(self.zero, old_cross_entropy) if flags.only_non_negative_entropy else old_cross_entropy
		# Stop gradient
		self.old_cross_entropy = tf.stop_gradient(self.old_cross_entropy)
		# Reduction function
		self.reduce_batch_function = eval('tf.reduce_{}'.format(flags.loss_type))
		self.ratio = self.get_ratio()
		
	def get(self, advantage):
		self.advantage = tf.stop_gradient(advantage)
		return eval('self.{}'.format(self.type))()
			
	def approximate_kullback_leibler_divergence(self): # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
		return self.reduce_batch_function(tf.keras.losses.MSE(self.old_cross_entropy, self.cross_entropy))
		
	def get_clipping_frequency(self):
		return tf.reduce_mean(tf.cast(tf.greater(tf.abs(self.ratio - self.one), self.clip_range), eval('tf.{}'.format(flags.parameters_type))))

	def get_importance_weight_batch(self):
		return tf.reduce_mean(self.ratio, -1)
		
	def get_entropy_regularization(self):
		if self.beta == 0:
			return self.zero
		return self.beta*self.reduce_batch_function(tf.reduce_mean(self.entropy, -1))
		
	def get_ratio(self):
		return tf.exp(self.old_cross_entropy - self.cross_entropy)
			
	def vanilla(self):
		gain = self.advantage * tf.reduce_sum(self.cross_entropy,axis=-1)
		return self.reduce_batch_function(gain)
		
	# Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
	def ppo(self):
		clipped_ratio = tf.clip_by_value(self.ratio, self.one-self.clip_range, self.one+self.clip_range)
		advantage = tf.expand_dims(self.advantage, 1)
		gain = tf.maximum(-advantage*self.ratio, -advantage*clipped_ratio)
		gain = tf.reduce_sum(gain, -1)
		return self.reduce_batch_function(gain)

	# Han, Seungyul, and Youngchul Sung. "Dimension-Wise Importance Sampling Weight Clipping for Sample-Efficient Reinforcement Learning." arXiv preprint arXiv:1905.02363 (2019).
	def disc(self):
		gain = self.ppo()
		# Compute IS loss function (Compute IS loss only for on-policy data) # In addition, in order to prevent the IS weight from being too far from 1, we include an additional loss for control
		kl_divergence = tf.cond(
			pred=self.is_replayed_batch[0], 
			true_fn=lambda: 0.,
			false_fn=lambda: self.approximate_kullback_leibler_divergence(),
		)
		# kl_divergence = self.approximate_kullback_leibler_divergence()
		# Update adaptive importance sampling target constant
		self.adaptive_importance_sampling_alpha = self.one
		self.adaptive_importance_sampling_alpha *= tf.cond(
			pred=kl_divergence > flags.importance_sampling_policy_target * 1.5, 
			true_fn=lambda: 2., # "Adaptive IS loss factor is increased"
			false_fn=lambda: 1.
		)
		self.adaptive_importance_sampling_alpha *= tf.cond(
			pred=kl_divergence < flags.importance_sampling_policy_target / 1.5, 
			true_fn=lambda: 0.5, # "Adaptive IS loss factor is reduced"
			false_fn=lambda: 1.
		)
		self.adaptive_importance_sampling_alpha = tf.clip_by_value(self.adaptive_importance_sampling_alpha, 2**(-10),64)
		return gain + tf.stop_gradient(self.adaptive_importance_sampling_alpha) * kl_divergence
