# -*- coding: utf-8 -*-
from agent.algorithm.rl_algorithm import RL_Algorithm
import tensorflow.compat.v1 as tf
from tf_agents.utils import common
from agent.network import is_continuous_control
from utils.distributions import Normal
from agent.algorithm.advantage_based.loss.policy_loss import PolicyLoss
import numpy as np
#===============================================================================
# from utils.running_std import RunningMeanStd
#===============================================================================
import options
flags = options.get()

def merge_splitted_advantages(advantage):
	if flags.split_values:
		return flags.extrinsic_coefficient*advantage[0] + flags.intrinsic_coefficient*advantage[1]
	return advantage

# TD3's original paper: https://arxiv.org/pdf/1802.09477.pdf
class TD3_Algorithm(RL_Algorithm): # taken from here: https://github.com/hill-a/stable-baselines
	has_td_error = True
	is_on_policy = False

	def __init__(self, group_id, model_id, environment_info, beta=None, training=True, parent=None, sibling=None, with_intrinsic_reward=True):
		self.train_step = 0
		self.setup(environment_info)
		super().__init__(group_id, model_id, environment_info, beta, training, parent, sibling, with_intrinsic_reward)

	def setup(self, environment_info):
		# Regularisation
		self.critic_regularisation_weight = 1e-2
		self.actor_regularisation_weight = 1e-2
		# Parameters updates
		self.target_update_period = 5
		self.actor_update_period = 10
		self.target_update_tau = 0.05
		# Action noise
		self.is_stochastic = False
		if not self.is_stochastic:
			self.exploration_noise_std = 0.05
			self.target_policy_noise = 0.1
			self.target_policy_noise_clip = 0.5
			self.exploration_noise_generator = [
				common.OUProcess(tf.zeros(head[0]), stddev=self.exploration_noise_std)
				for head in environment_info['action_shape']
			]
			self.target_noise_generator = [
				lambda: tf.clip_by_value(Normal(0., self.target_policy_noise).sample(), -self.target_policy_noise_clip, self.target_policy_noise_clip)
				for head in environment_info['action_shape']
			]

	def get_main_network_partitions(self):
		return [
			['Actor','TargetActor'],
			['Critic','TargetCritic'],
		]

	def build_fetch_maps(self):
		self.feed_map = {
			'states': self.state_batch,
			'new_states': self.new_state_batch,
			'policies': self.old_policy_batch,
			'actions': self.old_action_batch,
			'action_masks': self.old_action_mask_batch if self.has_masked_actions else None,
			'state_mean': self.state_mean_batch,
			'state_std': self.state_std_batch,
			'sizes': self.size_batch,
			'rewards': self.reward_batch,
			'terminal': self.terminal_batch,
		}
		self.fetch_map = {
			'actions': self.noisy_action_batch, 
			'hot_actions': self.action_batch, 
			'policies': self.policy_batch, 
			'values': self.state_value_batch, 
			'new_internal_states': self._get_internal_state() if flags.network_has_internal_state else None,
			'importance_weights': None,
			'extracted_relations': self.relations_sets if self.network['Actor'].produce_explicit_relations else None,
			'intrinsic_rewards': self.intrinsic_reward_batch if self.with_intrinsic_reward else None,
			'td_errors': self.td_error_batch,
		}

	def build_network(self):
		main_actor, main_critic = self.network['Actor'], self.network['Critic']
		target_actor, target_critic = self.network['TargetActor'], self.network['TargetCritic']
		####################################
		# [Intrinsic Rewards]
		if self.with_intrinsic_reward:
			reward_network_output = self.network['Reward'].build_embedding({
				'new_state': self.new_state_batch, 
				'state_mean': self.state_mean_batch,
				'state_std': self.state_std_batch,
			})
			self.intrinsic_reward_batch, intrinsic_reward_loss, self.training_state = reward_network_output
			print( "	[{}]Intrinsic Reward shape: {}".format(self.id, self.intrinsic_reward_batch.get_shape()) )
			print( "	[{}]Training State Kernel shape: {}".format(self.id, self.training_state['kernel'].get_shape()) )
			print( "	[{}]Training State Bias shape: {}".format(self.id, self.training_state['bias'].get_shape()) )		
		else:
			self.training_state = None
		####################################
		# [Model]
		def add_noise(action, noise):
			noisy_action = action + noise
			noisy_action = tf.where(
				tf.greater(noisy_action,1), 
				action - noise, 
				noisy_action
			)
			noisy_action = tf.where(
				tf.less(noisy_action,-1), 
				action - noise, 
				noisy_action
			)
			return tf.clip_by_value(noisy_action, -1,1)
		# Create the policy
		self.policy_batch = main_actor.policy_layer(
			input=main_actor.build_embedding(
				{
					'state': self.state_batch, 
					'size': self.size_batch,
				}, 
				use_internal_state=flags.network_has_internal_state, 
			), 
		)
		if self.is_stochastic:
			self.action_batch,self.noisy_action_batch,_ = self.sample_actions(self.policy_batch)
		else:
			self.action_batch,_,_ = self.sample_actions(self.policy_batch, mean=True)
			self.noisy_action_batch = [
				add_noise(action, noise_generator())
				for action,noise_generator in zip(self.action_batch, self.exploration_noise_generator)
			]
		# Use two Q-functions to improve performance by reducing overestimation bias
		main_embedding_and_old_action = main_critic.build_embedding(
			{
				'state': self.state_batch + self.old_action_batch, 
				'size': self.size_batch,
			}, 
			use_internal_state=flags.network_has_internal_state, 
		)
		main_embedding_and_new_action = main_critic.build_embedding(
			{
				'state': self.state_batch + self.action_batch, 
				'size': self.size_batch,
			}, 
			use_internal_state=flags.network_has_internal_state, 
		)
		q_value_1 = main_critic.value_layer(
			name='q_value_1', 
			input=main_embedding_and_old_action, 
		)
		q_value_2 = main_critic.value_layer(
			name='q_value_2', 
			input=main_embedding_and_old_action, 
		)
		# Q value when following the current policy
		q_value_on_policy_1 = main_critic.value_layer(
			name='q_value_1', # reusing q_value_1 net
			input=main_embedding_and_new_action, 
		)
		q_value_on_policy_2 = main_critic.value_layer(
			name='q_value_2', # reusing q_value_2 net
			input=main_embedding_and_new_action, 
		)
		self.state_value_batch = tf.minimum(q_value_on_policy_1,q_value_on_policy_2) # state_value_batch must not depend on old action
		self.q_value_1 = q_value_1[...,0]
		self.q_value_2 = q_value_2[...,0]
		self.q_value_pi_1 = q_value_on_policy_1[...,0]
		self.q_value_pi_2 = q_value_on_policy_2[...,0]
		print( "	[{}]Value output shape: {}".format(self.id, self.state_value_batch.get_shape()) )
		####################################
		# [Target]
		# Create target networks
		target_policy_batch = target_actor.policy_layer(
			input=target_actor.build_embedding(
				{
					'state': self.new_state_batch, 
					'size': self.size_batch,
				}, 
				use_internal_state=flags.network_has_internal_state, 
			), 
		)
		# Target policy smoothing, by adding clipped noise to target actions
		if self.is_stochastic:
			target_action_batch,noisy_target_action_batch,_ = self.sample_actions(target_policy_batch)
		else:
			target_action_batch,_,_ = self.sample_actions(target_policy_batch, mean=True)
			noisy_target_action_batch = [
				add_noise(action, noise_generator())
				for action,noise_generator in zip(target_action_batch, self.target_noise_generator)
			]
		# Q values when following the target policy
		target_embedding_and_old_action = target_critic.build_embedding(
			{
				'state': self.new_state_batch + target_action_batch, 
				'size': self.size_batch,
			}, 
			use_internal_state=flags.network_has_internal_state, 
		)
		target_embedding_and_new_action = target_critic.build_embedding(
			{
				'state': self.new_state_batch + noisy_target_action_batch, 
				'size': self.size_batch,
			}, 
			use_internal_state=flags.network_has_internal_state, 
		)	
		q_target_value_1 = target_critic.value_layer(
			name='q_target_value_1', 
			input=target_embedding_and_new_action, 
		)
		q_target_value_2 = target_critic.value_layer(
			name='q_target_value_2', 
			input=target_embedding_and_new_action, 
		)
		if self.value_count < 2:
			reward = tf.reduce_sum(self.reward_batch, axis=-1, keepdims=True)
		else:
			reward = self.reward_batch
		print( "	[{}]Reward shape: {}".format(self.id, reward.get_shape()) )
		# Take the min of the two target Q-Values (clipped Double-Q Learning)
		min_q_target_value = tf.minimum(q_target_value_1, q_target_value_2)
		print( "	[{}]Min. QTarget Value shape: {}".format(self.id, min_q_target_value.get_shape()) )
		# Targets for Q value regression
		discounted_q_value = tf.where(
			self.terminal_batch, 
			tf.zeros_like(min_q_target_value), 
			flags.extrinsic_gamma * min_q_target_value
		)
		print( "	[{}]Discounted QValue shape: {}".format(self.id, discounted_q_value.get_shape()) )
		td_target = tf.stop_gradient(reward + discounted_q_value)
		print( "	[{}]TD-Target shape: {}".format(self.id, td_target.get_shape()) )
		td_error_1 = common.element_wise_huber_loss(td_target, q_value_1)
		td_error_2 = common.element_wise_huber_loss(td_target, q_value_2)
		print( "	[{}]TD-Error 1 shape: {}".format(self.id, td_error_1.get_shape()) )
		print( "	[{}]TD-Error 2 shape: {}".format(self.id, td_error_2.get_shape()) )
		self.td_error_batch = td_target - self.state_value_batch
		print( "	[{}]Advantage shape: {}".format(self.id, self.td_error_batch.get_shape()) )
		####################################
		# [Relations sets]
		self.relations_sets = main_actor.relations_sets if main_actor.produce_explicit_relations else None
		####################################
		# [Loss]
		self._loss_builder = {}
		if self.with_intrinsic_reward:
			self._loss_builder['Reward'] = lambda global_step: (intrinsic_reward_loss,)
		def get_critic_loss(global_step): # Compute Q-Function loss
			with tf.variable_scope("critic_loss", reuse=False):
				critic_loss = tf.reduce_mean(td_error_1 + td_error_2)
				if self.critic_regularisation_weight > 0:
					critic_loss += self.get_regularisation_loss(partition_list=['Critic'], regularisation_weight=self.critic_regularisation_weight, loss_type=tf.nn.l2_loss)
				return critic_loss
		def get_actor_loss(global_step):
			with tf.variable_scope("actor_loss", reuse=False):
				qvalue_loss = -self.state_value_batch
				if self.value_count > 1:
					qvalue_loss = tf.expand_dims(tf.map_fn(fn=merge_splitted_advantages, elems=qvalue_loss), -1)
				qvalue_loss = tf.reduce_mean(qvalue_loss)
				if self.actor_regularisation_weight > 0:
					qvalue_loss += self.get_regularisation_loss(partition_list=['Actor'], regularisation_weight=self.actor_regularisation_weight, loss_type=tf.nn.l2_loss)
				if self.constrain_replay:
					constrain_loss = sum(
						tf.reduce_mean(
							tf.maximum(
								0., 
								Normal(action, self.exploration_noise_std).cross_entropy(tf.stop_gradient(old_action))
							)
						)
						for action,old_action in zip(self.action_batch, self.old_action_batch)
					)
					qvalue_loss += tf.cond(
						pred=self.is_replayed_batch[0], 
						true_fn=lambda: constrain_loss,
						false_fn=lambda: tf.constant(0., dtype=self.parameters_type)
					)
				if not self.is_stochastic:
					return qvalue_loss

				advantage = tf.stop_gradient(self.td_error_batch)
				if self.value_count > 1:
					advantage = tf.expand_dims(tf.map_fn(fn=merge_splitted_advantages, elems=advantage), -1)
				policy_builder = PolicyLoss(
					global_step= global_step,
					type= flags.policy_loss,
					beta= self.beta,
					policy_heads= self.policy_heads, 
					actor_batch= self.policy_batch,
					old_policy_batch= self.old_policy_batch, 
					old_action_batch= self.old_action_batch, 
					is_replayed_batch= self.is_replayed_batch,
					old_action_mask_batch= self.old_action_mask_batch if self.has_masked_actions else None,
				)
				self.importance_weight_batch = policy_builder.get_importance_weight_batch()
				print( "	[{}]Importance Weight shape: {}".format(self.id, self.importance_weight_batch.get_shape()) )
				self.policy_kl_divergence = policy_builder.approximate_kullback_leibler_divergence()
				self.policy_clipping_frequency = policy_builder.get_clipping_frequency()
				self.policy_entropy_regularization = policy_builder.get_entropy_regularization()
				policy_loss = policy_builder.get(advantage)
				# [Entropy regularization]
				if not flags.intrinsic_reward and flags.entropy_regularization:
					policy_loss += -self.policy_entropy_regularization
				return policy_loss + qvalue_loss

		self._loss_builder['Actor'] = lambda global_step: (get_actor_loss(global_step),)
		self._loss_builder['Critic'] = lambda global_step: (get_critic_loss(global_step),)
		####################################
		# [Params]
		# Q Values and policy target params
		source_params = self.network['Actor'].shared_keys + self.network['Critic'].shared_keys
		target_params = self.network['TargetActor'].shared_keys + self.network['TargetCritic'].shared_keys
		# Polyak averaging for target variables
		self.target_ops = common.soft_variables_update(source_params, target_params, tau=self.target_update_tau)
		# Initializing target to match source variables
		self.target_init_op = common.soft_variables_update(source_params, target_params, tau=1)

	def _train(self, feed_dict, replay=False):
		if self.train_step == 0:
			tf.get_default_session().run(self.target_init_op)
			self.sync()
		# Build _train fetches
		train_tuple = self.train_operations_dict['Critic'] # update critic
		if self.train_step%self.actor_update_period == 0: # update actor
			train_tuple += self.train_operations_dict['Actor']
		if self.train_step%self.target_update_period == 0: # update targets
			train_tuple += (self.target_ops,)
		# else: # update only the critic
		# Do not replay intrinsic reward training otherwise it would start to reward higher the states distant from extrinsic rewards
		if self.with_intrinsic_reward and not replay:
			train_tuple += self.train_operations_dict['Reward']
		
		self.train_step += 1
		# Build fetch
		fetches = [train_tuple] # Minimize loss
		# Get loss values for logging
		fetches += [self.loss_dict['Actor'] + self.loss_dict['Critic']] if flags.print_loss else [()]
		# Debug info
		if flags.print_policy_info:
			policy_info = (self.q_value_1, self.q_value_2, self.q_value_pi_1, self.q_value_pi_2)
			if self.is_stochastic:
				policy_info += (self.policy_kl_divergence, self.policy_clipping_frequency, self.policy_entropy_regularization)
			fetches.append(policy_info)
		else:
			fetches.append(())
		# Intrinsic reward
		fetches += [self.loss_dict['Reward']] if self.with_intrinsic_reward else [()]
		# Run
		_, loss, policy_info, reward_info = tf.get_default_session().run(fetches=fetches, feed_dict=feed_dict)
		self.sync()
		# Build and return loss dict
		train_info = {}
		if flags.print_loss:
			train_info["loss_actor"], train_info["loss_critic"] = loss
			train_info["loss_total"] = train_info["loss_actor"] + train_info["loss_critic"]
		if flags.print_policy_info:
			train_info["q_value_1"],train_info["q_value_2"],train_info["q_value_pi_1"],train_info["q_value_pi_2"] = map(lambda x: np.mean(x), policy_info[:4])
			if self.is_stochastic:
				train_info["kl_divergence"],train_info["clipping_frequency"],train_info["entropy"] = policy_info[4:]
		if self.with_intrinsic_reward:
			train_info["intrinsic_reward_loss"] = reward_info
		# Build loss statistics
		if train_info:
			self._train_statistics.add(stat_dict=train_info, type='train{}_'.format(self.model_id))
		#=======================================================================
		# if self.loss_distribution_estimator.update([abs(train_info['loss_actor'])]):
		# 	self.actor_loss_is_too_small = self.loss_distribution_estimator.mean <= flags.loss_stationarity_range
		#=======================================================================
		return train_info

	def _build_train_feed(self, info_dict):
		feed_dict = self._get_multihead_feed(target=self.state_batch, source=info_dict['states'])
		# Internal State
		if flags.network_has_internal_state:
			feed_dict.update( self._get_internal_state_feed([info_dict['internal_state']]) )
			feed_dict.update( {self.size_batch: [len(info_dict['states'])]} )
		# New states
		feed_dict.update( self._get_multihead_feed(target=self.new_state_batch, source=info_dict['new_states']) )
		# Add replay boolean to feed dictionary
		feed_dict.update( {self.terminal_batch: info_dict['terminal']} )
		# Old Action
		feed_dict.update( self._get_multihead_feed(target=self.old_policy_batch, source=info_dict['policies']) )
		feed_dict.update( self._get_multihead_feed(target=self.old_action_batch, source=info_dict['actions']) )
		# Reward
		feed_dict.update( {self.reward_batch: info_dict['rewards']} )
		return feed_dict
