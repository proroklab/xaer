# -*- coding: utf-8 -*-
from agent.algorithm.rl_algorithm import RL_Algorithm
import tensorflow.compat.v1 as tf
from utils.distributions import Categorical, Normal
from agent.network import is_continuous_control
from agent.algorithm.qvalue_based.noise import *
#===============================================================================
# from utils.running_std import RunningMeanStd
#===============================================================================
import options
flags = options.get()

# TD3's original paper: https://arxiv.org/pdf/1802.09477.pdf
class TD3_Algorithm(RL_Algorithm): # taken from here: https://github.com/hill-a/stable-baselines
	has_td_error = True
	is_on_policy = False

	def __init__(self, group_id, model_id, environment_info, beta=None, training=True, parent=None, sibling=None, with_intrinsic_reward=True):
		self.policy_delay = 2
		self.reset_noise_step_gap = 100
		self.target_policy_noise = 0.2
		self.target_noise_clip = 0.5
		self.gamma = flags.extrinsic_gamma
		self.tau = 0.005
		super().__init__(group_id, model_id, environment_info, beta, training, parent, sibling, with_intrinsic_reward)
		self.action_noise = [
			OrnsteinUhlenbeckActionNoise(mean=np.zeros(head['size']), sigma=0.1 * np.ones(head['size']))
			for head in self.policy_heads
		]

	def fetch_info(self, fetch_label_list, info_dict):
		result_dict = super().fetch_info(fetch_label_list, info_dict)
		if 'actions' in result_dict:
			result_dict['actions'] = [
				[
					np.clip(sub_action + noise(), -1,1)
					for sub_action, noise in zip(action, self.action_noise)
				]
				for action in result_dict['actions']
			]
		if self.train_step%self.reset_noise_step_gap == 0: # update policy and target
			for a in self.action_noise:
				a.reset()
		return result_dict

	def get_main_network_partitions(self):
		return [
			['Actor','TargetActor'],
			['Critic','TargetCritic']
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
			'actions': self.action_batch, 
			'hot_actions': self.action_batch, 
			'policies': self.policy_batch, 
			'values': self.state_value_batch, 
			'new_internal_states': self._get_internal_state() if flags.network_has_internal_state else None,
			'importance_weights': None,
			'extracted_relations': self.relations_sets if self.network['Actor'].produce_explicit_relations else None,
			'intrinsic_rewards': self.intrinsic_reward_batch if self.with_intrinsic_reward else None,
			'td_errors': self.td_error_batch,
		}

	def _get_noisy_action(self, action_batch, stddev, noise_clip):
		new_action_batch = []
		for a in action_batch:
			# Target policy smoothing, by adding clipped noise to target actions
			noise = tf.random_normal(tf.shape(a), stddev=stddev)
			noise = tf.clip_by_value(noise, -noise_clip, noise_clip)
			# Clip the noisy action to remain in the bounds [-1, 1] (output of a tanh)
			new_action_batch.append(tf.clip_by_value(a + noise, -1, 1))
		return new_action_batch

	def sample_actions(self, actor_batch, mean=False):
		action_batch = []
		for h,actor_head in enumerate(actor_batch):
			if is_continuous_control(self.policy_heads[h]['depth']):
				new_policy_batch = tf.transpose(actor_head, [1, 0, 2])
				if not mean:
					sample_batch = Normal(new_policy_batch[0], new_policy_batch[1]).sample()
				else:
					sample_batch = new_policy_batch[0]
				action = tf.clip_by_value(sample_batch, -1,1, name='action_clipper')
				action_batch.append(action) # Sample action batch in forward direction, use old action in backward direction
			else: # discrete control
				distribution = Categorical(actor_head)
				action = distribution.sample(one_hot=False) # Sample action batch in forward direction, use old action in backward direction
				action_batch.append(action)
		# Give self esplicative name to output for easily retrieving it in frozen graph
		# tf.identity(action_batch, name="action")
		return action_batch

	def build_network(self):
		main_actor, main_critic = self.network['Actor'], self.network['Critic']
		target_actor, target_critic = self.network['TargetActor'], self.network['TargetCritic']
		batch_dict = {
			'state': self.state_batch, 
			'size': self.size_batch,
		}
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
			batch_dict['training_state'] = self.training_state
		####################################
		# [Model]
		def concat_action_list(embedding, action_list):
			action_list = [
				tf.keras.layers.Flatten()(action_list[h])
				for h,_ in enumerate(self.policy_heads)
			]
			action = tf.concat(action_list, -1)
			result = tf.concat([embedding, action], axis=-1)
			return result
		# Create the policy
		self.policy_batch = main_actor.policy_layer(
			input=main_actor.build_embedding(
				batch_dict, 
				use_internal_state=flags.network_has_internal_state, 
				scope='MainPolicy'
			), 
			scope=main_actor.format_scope_name([main_actor.scope_name,'MainPolicy'])
		)
		self.action_batch = self.sample_actions(self.policy_batch, mean=True)
		# Use two Q-functions to improve performance by reducing overestimation bias
		main_embedding = main_critic.build_embedding(
			batch_dict, 
			use_internal_state=flags.network_has_internal_state, 
			scope='MainCritic'
		)
		main_embedding_and_action = concat_action_list(main_embedding, self.old_action_batch)
		q_value_1 = main_critic.value_layer(
			name='q_value_1', 
			input=main_embedding_and_action, 
			scope=main_critic.format_scope_name([main_critic.scope_name,'MainCritic'])
		)
		q_value_2 = main_critic.value_layer(
			name='q_value_2', 
			input=main_embedding_and_action, 
			scope=main_critic.format_scope_name([main_critic.scope_name,'MainCritic'])
		)
		# Q value when following the current policy
		q_value_on_policy = main_critic.value_layer(
			name='q_value_1', # reusing q_value_1 net
			input=concat_action_list(main_embedding, self.action_batch), 
			scope=main_critic.format_scope_name([main_critic.scope_name,'MainCritic'])
		)
		self.state_value_batch = q_value_on_policy #= tf.clip_by_value(tf.keras.activations.relu(q_value_on_policy), 0,1)
		print( "	[{}]Value output shape: {}".format(self.id, self.state_value_batch.get_shape()) )
		####################################
		# [Target]
		# Create target networks
		batch_dict['state'] = self.new_state_batch
		target_policy = target_actor.policy_layer(
			input=target_actor.build_embedding(
				batch_dict, 
				use_internal_state=flags.network_has_internal_state, 
				scope='TargetPolicy'
			), 
			scope=target_actor.format_scope_name([target_actor.scope_name,'TargetPolicy'])
		)
		# Target policy smoothing, by adding clipped noise to target actions
		target_action = self.sample_actions(target_policy, mean=True)
		# noisy_target_action = self.sample_actions(target_policy)
		noisy_target_action = self._get_noisy_action(target_action, self.target_policy_noise, self.target_noise_clip)
		# Q values when following the target policy
		target_embedding = target_critic.build_embedding(
			batch_dict, 
			use_internal_state=flags.network_has_internal_state, 
			scope='TargetCritic'
		)
		target_embedding_and_action = concat_action_list(target_embedding, noisy_target_action)
		q_target_value_1 = target_critic.value_layer(
			name='q_target_value_1', 
			input=target_embedding_and_action, 
			scope=target_critic.format_scope_name([target_critic.scope_name,'TargetCritic'])
		)
		q_target_value_2 = target_critic.value_layer(
			name='q_target_value_2', 
			input=target_embedding_and_action, 
			scope=target_critic.format_scope_name([target_critic.scope_name,'TargetCritic'])
		)
		target_critic.value_layer( # this is used only to perfectly mirror the main network
			name='q_target_value_1', # reusing q_value_1 net
			input=concat_action_list(target_embedding, target_action), 
			scope=target_critic.format_scope_name([target_critic.scope_name,'TargetCritic'])
		)
		# Take the min of the two target Q-Values (clipped Double-Q Learning)
		min_q_target_value = tf.minimum(q_target_value_1, q_target_value_2)
		# Targets for Q value regression
		rewards = self.reward_batch if self.value_count==2 else tf.reduce_sum(self.reward_batch, -1)
		discounted_q_value = tf.where(self.terminal_batch, tf.zeros_like(min_q_target_value), self.gamma * min_q_target_value)
		discounted_q_value = tf.stop_gradient(discounted_q_value)
		td_target = rewards + discounted_q_value
		self.td_error_batch = -tf.abs(td_target - q_value_1)
		####################################
		# [Relations sets]
		self.relations_sets = main_actor.relations_sets if main_actor.produce_explicit_relations else None
		####################################
		# [Loss]
		self._loss_builder = {}
		if self.with_intrinsic_reward:
			self._loss_builder['Reward'] = lambda global_step: (intrinsic_reward_loss,)
		def get_critic_loss(global_step): # Compute Q-Function loss
			return tf.reduce_mean((td_target - q_value_1)**2) + tf.reduce_mean((td_target - q_value_2)**2)
		def get_actor_loss(global_step):
			# clipped_q_value = tf.clip_by_value(q_value_on_policy, tf.stop_gradient(q_value_1*(1-0.2)), tf.stop_gradient(q_value_1*(1+0.2)))
			return -tf.reduce_mean(q_value_on_policy) # Policy loss: maximise q value
		self._loss_builder['Actor'] = lambda global_step: (get_actor_loss(global_step),)
		self._loss_builder['Critic'] = lambda global_step: (get_critic_loss(global_step),)
		####################################
		# [Params]
		# Q Values and policy target params
		source_params = self.network['Actor'].shared_keys + self.network['Critic'].shared_keys
		target_params = self.network['TargetActor'].shared_keys + self.network['TargetCritic'].shared_keys
		# Polyak averaging for target variables
		self.target_ops = [
			tf.assign(target, (1 - self.tau) * target + self.tau * source)
			for target, source in zip(target_params, source_params)
		]
		# Initializing target to match source variables
		self.target_init_op = [
			tf.assign(target, source)
			for target, source in zip(target_params, source_params)
		]
		self.train_step = 0

	def _train(self, feed_dict, replay=False):
		# Build _train fetches
		train_tuple = (self.train_operations_dict['Critic'],)
		# Do not replay intrinsic reward training otherwise it would start to reward higher the states distant from extrinsic rewards
		if self.with_intrinsic_reward and not replay:
			train_tuple += (self.train_operations_dict['Reward'],)
		if self.train_step == 0:
			train_tuple += (self.train_operations_dict['Actor'], self.target_init_op,)
		elif self.train_step%self.policy_delay == 0: # update policy and target
			train_tuple += (self.train_operations_dict['Actor'], self.target_ops,)
		self.train_step += 1
		# Build fetch
		fetches = [train_tuple] # Minimize loss
		# Get loss values for logging
		fetches += [(self.loss_dict['Actor'],self.loss_dict['Critic'])] if flags.print_loss else [()]
		# Debug info
		fetches += [()]
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
		feed_dict.update( self._get_multihead_feed(target=self.old_action_batch, source=info_dict['actions']) )
		# Reward
		feed_dict.update( {self.reward_batch: info_dict['rewards']} )
		return feed_dict
