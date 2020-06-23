# -*- coding: utf-8 -*-
from agent.algorithm.rl_algorithm import RL_Algorithm
import tensorflow.compat.v1 as tf
from agent.algorithm.advantage_based.loss.policy_loss import PolicyLoss
from agent.algorithm.advantage_based.loss.value_loss import ValueLoss
from utils.distributions import Categorical, Normal
from agent.network import is_continuous_control
from agent.algorithm.qvalue_based.noise import *
#===============================================================================
# from utils.running_std import RunningMeanStd
#===============================================================================
import options
flags = options.get()

class TD3_Algorithm(RL_Algorithm):

	def __init__(self, group_id, model_id, environment_info, beta=None, training=True, parent=None, sibling=None, with_intrinsic_reward=True):
		super().__init__(group_id, model_id, environment_info, beta, training, parent, sibling, with_intrinsic_reward)
		self.action_noise = [
			OrnsteinUhlenbeckActionNoise(mean=np.zeros(head['size']), sigma=0.1 * np.ones(head['size']))
			for head in self.policy_heads
		]
		self.target_policy_noise = 0.2
		self.gamma = flags.gamma
		self.tau = 0.005
		self.policy_delay = 2

	def get_main_network_partitions(self):
		return [
			['ActorCritic'],
			['TargetActorCritic']
		]

	def build_network(self):
		main_net = self.network['ActorCritic']
		target_net = self.network['TargetActorCritic']
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
		# Create the policy
		embedded_input = main_net.build_embedding(batch_dict, use_internal_state=flags.network_has_internal_state, name='ActorCritic')
		self.policy_out = policy_out = main_net.policy_layer(
			input=embedded_input, 
			scope=main_net.scope_name
		)
		def concat_action(embedding, action):
			action = list(map(tf.layers.flatten,action))
			action = tf.transpose(action, [1,0,2])
			action = tf.layers.flatten(action)
			return = tf.concat([embedding, action], axis=-1)
		# Use two Q-functions to improve performance by reducing overestimation bias
		old_qf = concat_action(embedded_input, self.old_action_batch)
		qf1 = main_net.value_layer(name='qf1', input=old_qf, scope=main_net.scope_name)
		qf2 = main_net.value_layer(name='qf2', input=old_qf, scope=main_net.scope_name)
		# Q value when following the current policy
		qf1_pi = main_net.value_layer(
			name='qf1', # reusing qf1 net
			input=concat_action(embedded_input, policy_out), 
			scope=main_net.scope_name
		)
		####################################
		# [Target]
		# Create target networks
		target_embedded_input = target_net.build_embedding(batch_dict, use_internal_state=flags.network_has_internal_state, name='TargetActorCritic')
		batch_dict['state'] = self.new_state_batch
		target_policy_out = target_net.policy_layer(
			input=target_embedded_input, 
			scope=target_net.scope_name
		)
		# Target policy smoothing, by adding clipped noise to target actions
		target_noise = tf.random_normal(tf.shape(target_policy_out), stddev=self.target_policy_noise)
		target_noise = tf.clip_by_value(target_noise, -self.target_noise_clip, self.target_noise_clip)
		# Clip the noisy action to remain in the bounds [-1, 1] (output of a tanh)
		noisy_target_action = tf.clip_by_value(target_policy_out + target_noise, -1, 1)
		# Q values when following the target policy
		noisy_qf = concat_action(target_embedded_input, noisy_target_action)
		qf1_target = target_net.value_layer(input=noisy_qf, scope=target_net.scope_name, name='qf1_target')
		qf2_target = target_net.value_layer(input=noisy_qf, scope=target_net.scope_name, name='qf2_target')
		# Take the min of the two target Q-Values (clipped Double-Q Learning)
		self.state_value_batch = min_qf_target = tf.minimum(qf1_target, qf2_target)
		####################################
		# [Relations sets]
		self.relations_sets = main_net.relations_sets if main_net.produce_explicit_relations else None
		####################################
		# [Loss]
		self._loss_builder = {}
		if self.with_intrinsic_reward:
			self._loss_builder['Reward'] = lambda global_step: (intrinsic_reward_loss,)
		def get_critic_loss(global_step):
			# Targets for Q value regression
			q_backup = tf.stop_gradient(
				self.reward_batch +
				tf.cast(not self.terminal_batch, tf.float32) * self.gamma * min_qf_target
			)

			# Compute Q-Function loss
			qf1_loss = tf.reduce_mean((q_backup - qf1) ** 2)
			qf2_loss = tf.reduce_mean((q_backup - qf2) ** 2)

			return qf1_loss + qf2_loss
		def get_actor_loss(global_step):
			return -tf.reduce_mean(qf1_pi) # Policy loss: maximise q value
		self._loss_builder['ActorCritic'] = lambda global_step: (get_actor_loss(global_step), get_critic_loss(global_step))
		####################################
		# [Params]
		# Q Values and policy target params
		source_params = self.network['ActorCritic'].shared_keys
		target_params = self.network['TargetActorCritic'].shared_keys
		# Polyak averaging for target variables
		self.target_ops = [
			tf.assign(target, (1 - self.tau) * target + self.tau * source)
			for target, source in zip(target_params, source_params)
		]
		self.train_step = 0

	def predict_action(self, info_dict):
		batch_size = info_dict['sizes']
		batch_count = len(batch_size)
		# State
		feed_dict = self._get_multihead_feed(target=self.state_batch, source=info_dict['states'])
		# Internal state
		if flags.network_has_internal_state:
			feed_dict.update( self._get_internal_state_feed( info_dict['internal_states'] ) )
			feed_dict.update( {self.size_batch: batch_size} )
		# Return action_batch, policy_batch, new_internal_state
		policy_batch, action_batch, value_batch, new_internal_states = tf.get_default_session().run(
			fetches=[
				self.policy_out, 
				self.noisy_policy_out, 
				self.state_value_batch, 
				self._get_internal_state(),
			], 
			feed_dict=feed_dict
		)
		# Properly format for output the internal state
		new_internal_states = self._format_internal_state(new_internal_states, batch_count)
		# Properly format for output: action and policy may have multiple heads, swap 1st and 2nd axis
		action_batch = tuple(zip(*action_batch))
		hot_action_batch = tuple(zip(*action_batch))
		policy_batch = tuple(zip(*policy_batch))
		# Return output
		return action_batch, hot_action_batch, policy_batch, value_batch, new_internal_states

	def _train(self, feed_dict, replay=False):
		if self.train_step == 0:
			train_tuple += (self.target_ops,)
		self.train_step = (self.train_step + 1)%self.policy_delay
		# Build _train fetches
		train_tuple = (self.train_operations_dict['ActorCritic'],)
		# Do not replay intrinsic reward training otherwise it would start to reward higher the states distant from extrinsic rewards
		if self.with_intrinsic_reward and not replay:
			train_tuple += (self.train_operations_dict['Reward'],)
		# Build fetch
		fetches = [train_tuple] # Minimize loss
		# Get loss values for logging
		fetches += [self.loss_dict['ActorCritic']] if flags.print_loss else [()]
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
			train_info["loss_total"] = sum(loss)
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
		if self.with_intrinsic_reward:
			feed_dict.update( self._get_multihead_feed(target=self.new_state_batch, source=info_dict['new_states']) )
		# Internal State
		if flags.network_has_internal_state:
			feed_dict.update( self._get_internal_state_feed([info_dict['internal_state']]) )
			feed_dict.update( {self.size_batch: [len(info_dict['cumulative_returns'])]} )
		# New states
		feed_dict.update( self._get_multihead_feed(target=self.new_state_batch, source=info_dict['new_states']) )
		# Add replay boolean to feed dictionary
		feed_dict.update( {self.terminal_batch: [info_dict['terminal']]} )
		# Old Action
		feed_dict.update( self._get_multihead_feed(target=self.old_action_batch, source=info_dict['actions']) )
		# Reward
		feed_dict.update( {self.reward_batch: [info_dict['rewards']]} )
		return feed_dict
