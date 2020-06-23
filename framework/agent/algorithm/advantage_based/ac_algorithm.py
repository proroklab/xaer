# -*- coding: utf-8 -*-
from agent.algorithm.rl_algorithm import RL_Algorithm
import tensorflow.compat.v1 as tf
from agent.algorithm.advantage_based.loss.policy_loss import PolicyLoss
from agent.algorithm.advantage_based.loss.value_loss import ValueLoss
from utils.distributions import Categorical, Normal
from agent.network import is_continuous_control
from agent.algorithm.advantage_based.advantage_estimator import *
#===============================================================================
# from utils.running_std import RunningMeanStd
#===============================================================================
import options
flags = options.get()

def merge_splitted_advantages(advantage):
	return flags.extrinsic_coefficient*advantage[0] + flags.intrinsic_coefficient*advantage[1]

class AC_Algorithm(RL_Algorithm):
	extract_importance_weight = flags.advantage_estimator.lower() in ["vtrace","gae_v"]

	def __init__(self, group_id, model_id, environment_info, beta=None, training=True, parent=None, sibling=None, with_intrinsic_reward=True):
		super().__init__(group_id, model_id, environment_info, beta, training, parent, sibling, with_intrinsic_reward)
		self.train_critic_when_replaying = flags.train_critic_when_replaying

	@staticmethod
	def get_reversed_cumulative_return(gamma, last_value, reversed_reward, reversed_value, reversed_extra, reversed_importance_weight):
		return eval(flags.advantage_estimator.lower())(
			gamma=gamma, 
			last_value=last_value, 
			reversed_reward=reversed_reward, 
			reversed_value=reversed_value, 
			reversed_extra=reversed_extra, 
			reversed_importance_weight=reversed_importance_weight,
		)

	def get_main_network_partitions(self):
		return [['ActorCritic']]

	def build_network(self):
		net = self.network['ActorCritic']
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
		# [Actor]
		embedding = net.build_embedding(batch_dict, use_internal_state=flags.network_has_internal_state, name='ActorCritic')
		self.actor_batch = net.policy_layer(
			input=embedding, 
			scope=net.scope_name
		)
		for i,b in enumerate(self.actor_batch): 
			print( "	[{}]Actor{} output shape: {}".format(self.id, i, b.get_shape()) )
		self.action_batch, self.hot_action_batch = self.sample_actions(self.actor_batch)
		for i,b in enumerate(self.action_batch): 
			print( "	[{}]Action{} output shape: {}".format(self.id, i, b.get_shape()) )
		for i,b in enumerate(self.hot_action_batch): 
			print( "	[{}]HotAction{} output shape: {}".format(self.id, i, b.get_shape()) )
		####################################
		# [Critic]
		self.state_value_batch = net.value_layer(
			input=embedding, 
			scope=net.scope_name
		)
		print( "	[{}]Critic output shape: {}".format(self.id, self.state_value_batch.get_shape()) )
		####################################
		# [Relations sets]
		self.relations_sets = net.relations_sets if net.produce_explicit_relations else None
		####################################
		# [Loss]
		self._loss_builder = {}
		if self.with_intrinsic_reward:
			self._loss_builder['Reward'] = lambda global_step: (intrinsic_reward_loss,)
		def get_actor_loss(global_step):
			with tf.variable_scope("actor_loss", reuse=False):
				print( "Preparing Actor loss {}".format(self.id) )
				policy_builder = PolicyLoss(
					global_step= global_step,
					type= flags.policy_loss,
					beta= self.beta,
					policy_heads= self.policy_heads, 
					actor_batch= self.actor_batch,
					old_policy_batch= self.old_policy_batch, 
					old_action_batch= self.old_action_batch, 
					is_replayed_batch= self.is_replayed_batch,
					old_action_mask_batch= self.old_action_mask_batch if self.has_masked_actions else None,
				)
				# if flags.runtime_advantage:
				# 	self.advantage_batch = adv = self.cumulative_return_batch - self.state_value_batch # baseline is always up to date
				self.importance_weight_batch = policy_builder.get_importance_weight_batch()
				print( "	[{}]Importance Weight shape: {}".format(self.id, self.importance_weight_batch.get_shape()) )
				self.policy_kl_divergence = policy_builder.approximate_kullback_leibler_divergence()
				self.policy_clipping_frequency = policy_builder.get_clipping_frequency()
				self.policy_entropy_regularization = policy_builder.get_entropy_regularization()
				policy_loss = policy_builder.get(tf.map_fn(fn=merge_splitted_advantages, elems=self.advantage_batch) if self.value_count > 1 else self.advantage_batch)
				# [Entropy regularization]
				if not flags.intrinsic_reward and flags.entropy_regularization:
					policy_loss += -self.policy_entropy_regularization
				# [Constraining Replay]
				if self.constrain_replay:
					constrain_loss = sum(
						0.5*builder.reduce_function(tf.squared_difference(new_distribution.mean(), tf.stop_gradient(old_action))) 
						for builder, new_distribution, old_action in zip(policy_loss_builder, new_policy_distributions, self.old_action_batch)
					)
					policy_loss += tf.cond(
						pred=self.is_replayed_batch[0], 
						true_fn=lambda: constrain_loss,
						false_fn=lambda: tf.constant(0., dtype=self.parameters_type)
					)
				return policy_loss
		def get_critic_loss(global_step):
			with tf.variable_scope("critic_loss", reuse=False):
				loss = flags.value_coefficient * ValueLoss(
					global_step=global_step,
					loss=flags.value_loss,
					prediction=self.state_value_batch, 
					old_prediction=self.old_state_value_batch, 
					target=self.cumulative_return_batch
				).get()
				if self.train_critic_when_replaying:
					return loss
				return tf.cond(
					pred=self.is_replayed_batch[0], 
					true_fn=lambda: tf.constant(0., dtype=self.parameters_type),
					false_fn=lambda: loss
				)
		self._loss_builder['ActorCritic'] = lambda global_step: (get_actor_loss(global_step), get_critic_loss(global_step))

	def sample_actions(self, actor_batch):
		action_batch = []
		hot_action_batch = []
		for h,actor_head in enumerate(actor_batch):
			if is_continuous_control(self.policy_heads[h]['depth']):
				new_policy_batch = tf.transpose(actor_head, [1, 0, 2])
				sample_batch = Normal(new_policy_batch[0], new_policy_batch[1]).sample()
				action = tf.clip_by_value(sample_batch, -1,1, name='action_clipper')
				action_batch.append(action) # Sample action batch in forward direction, use old action in backward direction
				hot_action_batch.append(action)
			else: # discrete control
				distribution = Categorical(actor_head)
				action = distribution.sample(one_hot=False) # Sample action batch in forward direction, use old action in backward direction
				action_batch.append(action)
				hot_action_batch.append(distribution.get_sample_one_hot(action))
		# Give self esplicative name to output for easily retrieving it in frozen graph
		# tf.identity(action_batch, name="action")
		return action_batch, hot_action_batch
	
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
		action_batch, hot_action_batch, policy_batch, value_batch, new_internal_states = tf.get_default_session().run(
			fetches=[
				self.action_batch, 
				self.hot_action_batch, 
				self.actor_batch, 
				self.state_value_batch, 
				self._get_internal_state(),
			], 
			feed_dict=feed_dict
		)
		# Properly format for output the internal state
		if len(new_internal_states) == 0:
			new_internal_states = [new_internal_states]*batch_count
		else:
			new_internal_states = [
				[
					[
						sub_partition_new_internal_state[i]
						for sub_partition_new_internal_state in partition_new_internal_states
					]
					for partition_new_internal_states in new_internal_states
				]
				for i in range(batch_count)
			]
		# Properly format for output: action and policy may have multiple heads, swap 1st and 2nd axis
		action_batch = tuple(zip(*action_batch))
		hot_action_batch = tuple(zip(*hot_action_batch))
		policy_batch = tuple(zip(*policy_batch))
		# Return output
		return action_batch, hot_action_batch, policy_batch, value_batch, new_internal_states

	def get_importance_weight(self, info_dict):
		# State
		feed_dict = self._get_multihead_feed(target=self.state_batch, source=info_dict['states'])
		# Old Policy & Action
		feed_dict.update( self._get_multihead_feed(target=self.old_policy_batch, source=info_dict['policies']) )
		feed_dict.update( self._get_multihead_feed(target=self.old_action_batch, source=info_dict['actions']) )
		if self.has_masked_actions:
			feed_dict.update( self._get_multihead_feed(target=self.old_action_mask_batch, source=info_dict['action_masks']) )
		# Internal State
		if flags.network_has_internal_state:
			feed_dict.update( self._get_internal_state_feed(info_dict['internal_states']) )
			feed_dict.update( {self.size_batch: info_dict['sizes']} )
		# Return value_batch
		return tf.get_default_session().run(
			fetches=self.importance_weight_batch, 
			feed_dict=feed_dict
		)

	def get_extracted_relations(self, info_dict):
		# State
		feed_dict = self._get_multihead_feed(target=self.state_batch, source=info_dict['states'])
		# Return value_batch
		return tf.get_default_session().run(
			fetches=self.relations_sets, 
			feed_dict=feed_dict
		)
	
	def _train(self, feed_dict, replay=False):
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
		fetches += [(self.policy_kl_divergence, self.policy_clipping_frequency, self.policy_entropy_regularization)] if flags.print_policy_info else [()]
		# Intrinsic reward
		fetches += [self.loss_dict['Reward']] if self.with_intrinsic_reward else [()]
		# Run
		_, loss, policy_info, reward_info = tf.get_default_session().run(fetches=fetches, feed_dict=feed_dict)
		self.sync()
		# Build and return loss dict
		train_info = {}
		if flags.print_loss:
			train_info["loss_actor"],train_info["loss_critic"] = loss
			train_info["loss_total"] = sum(loss)
		if flags.print_policy_info:
			train_info["actor_kl_divergence"], train_info["actor_clipping_frequency"], train_info["actor_entropy"] = policy_info
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
		# State & Cumulative Return & Old Value
		feed_dict = {}
		feed_dict.update({self.cumulative_return_batch: info_dict['cumulative_returns']})
		# PVO
		if flags.value_loss.lower() == 'pvo':
			feed_dict.update({self.old_state_value_batch: info_dict['values']})
		feed_dict.update( self._get_multihead_feed(target=self.state_batch, source=info_dict['states']) )
		feed_dict.update( self._get_multihead_feed(target=self.new_state_batch, source=info_dict['new_states']) )
		# Advantage
		feed_dict.update( {self.advantage_batch: info_dict['advantages']} )
		# Old Policy & Action
		feed_dict.update( self._get_multihead_feed(target=self.old_policy_batch, source=info_dict['policies']) )
		feed_dict.update( self._get_multihead_feed(target=self.old_action_batch, source=info_dict['actions']) )
		if self.has_masked_actions:
			feed_dict.update( self._get_multihead_feed(target=self.old_action_mask_batch, source=info_dict['action_masks']) )
		# Internal State
		if flags.network_has_internal_state:
			feed_dict.update( self._get_internal_state_feed([info_dict['internal_state']]) )
			feed_dict.update( {self.size_batch: [len(info_dict['cumulative_returns'])]} )
		return feed_dict
