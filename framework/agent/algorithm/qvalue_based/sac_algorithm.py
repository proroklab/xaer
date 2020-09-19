# -*- coding: utf-8 -*-
from agent.algorithm.qvalue_based.td3_algorithm import *
from agent.network import Singular_Network
import numpy as np
#===============================================================================
# from utils.running_std import RunningMeanStd
#===============================================================================
import options
flags = options.get()

# SAC's original paper: https://arxiv.org/pdf/1801.01290.pdf
class SAC_Algorithm(TD3_Algorithm): # taken from here: https://github.com/hill-a/stable-baselines
	# target_update_period = 5
	# actor_update_period = 10
	# target_update_tau = 0.05

	def __init__(self, group_id, model_id, environment_info, beta=None, training=True, parent=None, sibling=None, with_intrinsic_reward=True):
		self.initial_log_alpha = 0.
		self.target_entropy = -sum(
			head[0]*(head[1] if len(head) > 1 else 1)
			for head in environment_info['action_shape']
		) / 2.0
		self.critic_loss_weight = 0.5
		self.actor_loss_weight = 1
		self.alpha_loss_weight = 1
		self.use_log_alpha_in_alpha_loss = True
		super().__init__(group_id, model_id, environment_info, beta, training, parent, sibling, with_intrinsic_reward)

	def get_main_network_partitions(self):
		return [
			['Actor'],
			['Critic','TargetCritic'],
		]

	def get_external_network_partitions(self):
		return super().get_external_network_partitions()+[
			['Alpha']
		]

	def initialize_network(self):
		super().initialize_network()
		self.network['Alpha'] = Singular_Network(id=self.id, name='Alpha', scope_dict={'self': f"Alpha{self.id}"}, training=self.training)

	def build_network(self):
		main_actor, main_critic = self.network['Actor'], self.network['Critic']
		target_critic = self.network['TargetCritic']
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
		self.log_alpha = self.network['Alpha'].get(self.initial_log_alpha)
		if self.value_count < 2:
			reward = tf.reduce_sum(self.reward_batch, axis=-1, keepdims=True)
		else:
			reward = self.reward_batch
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
		self.noisy_action_batch,self.action_batch,log_pi = self.sample_actions(self.policy_batch)
		log_pi = tf.add_n(log_pi)
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
		next_policy_batch = main_actor.policy_layer(
			input=main_actor.build_embedding(
				{
					'state': self.new_state_batch, 
					'size': self.size_batch,
				}, 
				use_internal_state=flags.network_has_internal_state, 
			), 
		)
		# Target policy smoothing, by adding clipped noise to target actions
		noisy_next_action_batch,_,next_log_pi = self.sample_actions(next_policy_batch)
		next_log_pi = tf.add_n(next_log_pi)
		# Q values when following the target policy
		target_next_embedding_and_new_action = target_critic.build_embedding(
			{
				'state': self.new_state_batch + noisy_next_action_batch, 
				'size': self.size_batch,
			}, 
			use_internal_state=flags.network_has_internal_state, 
		)	
		q_target_next_value_1 = target_critic.value_layer(
			name='q_target_next_value_1', 
			input=target_next_embedding_and_new_action, 
		)
		q_target_next_value_2 = target_critic.value_layer(
			name='q_target_next_value_2', 
			input=target_next_embedding_and_new_action, 
		)
		print( "	[{}]Reward shape: {}".format(self.id, reward.get_shape()) )
		# Take the min of the two target Q-Values (clipped Double-Q Learning)
		min_q_target_value = tf.minimum(q_target_next_value_1, q_target_next_value_2) - (tf.exp(self.log_alpha) * next_log_pi)
		print( "	[{}]Min. QTarget Value shape: {}".format(self.id, min_q_target_value.get_shape()) )
		# Targets for Q value regression
		discounted_q_value = tf.where(
			self.terminal_batch, 
			tf.zeros_like(min_q_target_value), 
			self.gamma * min_q_target_value
		)
		print( "	[{}]Discounted QValue shape: {}".format(self.id, discounted_q_value.get_shape()) )
		td_target = tf.stop_gradient(reward + discounted_q_value)
		print( "	[{}]TD-Target shape: {}".format(self.id, td_target.get_shape()) )
		td_error_1 = tf.math.squared_difference(td_target, q_value_1)
		td_error_2 = tf.math.squared_difference(td_target, q_value_2)
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
				critic_loss = td_error_1 + td_error_2
				print( "	[{}]Critic loss 1 shape: {}".format(self.id, critic_loss.get_shape()) )
				critic_loss = tf.reduce_sum(critic_loss, axis=-1)
				print( "	[{}]Critic loss 2 shape: {}".format(self.id, critic_loss.get_shape()) )
				critic_loss = tf.reduce_mean(critic_loss)
				print( "	[{}]Critic loss 3 shape: {}".format(self.id, critic_loss.get_shape()) )
				critic_loss *= self.critic_loss_weight
				if self.critic_regularisation_weight > 0:
					critic_loss += self.get_regularisation_loss(partition_list=['Critic'], regularisation_weight=self.critic_regularisation_weight, loss_type=tf.nn.l2_loss)
				return critic_loss
		def get_actor_loss(global_step):
			with tf.variable_scope("actor_loss", reuse=False):
				value = self.state_value_batch
				if self.value_count > 1:
					value = tf.expand_dims(tf.map_fn(fn=merge_splitted_advantages, elems=value), -1)
				qvalue_loss = (tf.exp(self.log_alpha) * log_pi) - value
				print( "	[{}]Value loss 1 shape: {}".format(self.id, qvalue_loss.get_shape()) )
				qvalue_loss = tf.reduce_sum(qvalue_loss, axis=-1)
				print( "	[{}]Value loss 2 shape: {}".format(self.id, qvalue_loss.get_shape()) )
				qvalue_loss = tf.reduce_mean(qvalue_loss)
				print( "	[{}]Value loss 3 shape: {}".format(self.id, qvalue_loss.get_shape()) )
				qvalue_loss *= self.actor_loss_weight
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
				return qvalue_loss
		def get_alpha_loss(global_step):
			entropy_diff = tf.stop_gradient(-log_pi - self.target_entropy)
			if self.use_log_alpha_in_alpha_loss:
				alpha_loss = self.log_alpha * entropy_diff
			else:
				alpha_loss = tf.exp(self.log_alpha) * entropy_diff
			return self.alpha_loss_weight * tf.reduce_mean(alpha_loss)

		self._loss_builder['Actor'] = lambda global_step: (get_actor_loss(global_step),)
		self._loss_builder['Critic'] = lambda global_step: (get_critic_loss(global_step),)
		self._loss_builder['Alpha'] = lambda global_step: (get_alpha_loss(global_step),)
		####################################
		# [Params]
		# Q Values and policy target params
		source_params = self.network['Critic'].shared_keys
		target_params = self.network['TargetCritic'].shared_keys
		# Polyak averaging for target variables
		self.target_ops = common.soft_variables_update(source_params, target_params, tau=SAC_Algorithm.target_update_tau)
		# Initializing target to match source variables
		self.target_init_op = common.soft_variables_update(source_params, target_params, tau=1)

	def _train(self, feed_dict, replay=False):
		if self.train_step == 0:
			tf.get_default_session().run(self.target_init_op)
			self.sync()
		# Build _train fetches
		train_tuple = self.train_operations_dict['Critic']+self.train_operations_dict['Alpha'] # update critic
		if self.train_step%SAC_Algorithm.actor_update_period == 0: # update actor
			train_tuple += self.train_operations_dict['Actor']
		if self.train_step%SAC_Algorithm.target_update_period == 0: # update targets
			train_tuple += (self.target_ops,)
		# else: # update only the critic
		# Do not replay intrinsic reward training otherwise it would start to reward higher the states distant from extrinsic rewards
		if self.with_intrinsic_reward and not replay:
			train_tuple += self.train_operations_dict['Reward']
		
		self.train_step += 1
		# Build fetch
		fetches = [train_tuple] # Minimize loss
		# Get loss values for logging
		fetches += [self.loss_dict['Actor']+self.loss_dict['Critic']+self.loss_dict['Alpha']] if flags.print_loss else [()]
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
			train_info["loss_actor"], train_info["loss_critic"], train_info["loss_alpha"] = loss
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
