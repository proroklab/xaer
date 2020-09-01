# -*- coding: utf-8 -*-
import utils.tensorflow_utils as tf_utils
import tensorflow.compat.v1 as tf
import itertools as it
from collections import deque
from utils.statistics import Statistics
from utils.misc import flatten
from utils.distributions import Categorical, Normal
from agent.network import *
#===============================================================================
# from utils.running_std import RunningMeanStd
#===============================================================================
import options
flags = options.get()

class RL_Algorithm(object):
	build_cumulative_return = False
	has_importance_weight = False
	extract_importance_weight = False
	has_td_error = False
	has_advantage = False
	is_on_policy = True

	def __init__(self, group_id, model_id, environment_info, beta=None, training=True, parent=None, sibling=None, with_intrinsic_reward=True):
		self.parameters_type = eval('tf.{}'.format(flags.parameters_type))
		self.beta = beta if beta is not None else flags.beta
		self.value_count = 2 if flags.split_values else 1
		# initialize
		self.training = training
		self.group_id = group_id
		self.model_id = model_id
		self.id = '{0}_{1}'.format(self.group_id,self.model_id) # model id
		self.parent = parent if parent is not None else self # used for sharing with other models in hierarchy, if any
		self.sibling = sibling if sibling is not None else self # used for sharing with other models in hierarchy, if any
		# Environment info
		action_shape = environment_info['action_shape']
		self.policy_heads = [
			{
				'size':head[0], # number of actions to take
				'depth':head[1] if len(head) > 1 else 0 # number of discrete action types: set 0 for continuous control
			}
			for head in action_shape
		]
		state_shape = environment_info['state_shape']
		self.state_heads = [
			{'shape':head}
			for head in state_shape
		]
		self.state_scaler = environment_info['state_scaler'] # state scaler, for saving memory (eg. in case of RGB input: uint8 takes less memory than float64)
		self.has_masked_actions = environment_info['has_masked_actions']
		# Create the network
		self.with_intrinsic_reward = with_intrinsic_reward
		self.build_input_placeholders()
		self.initialize_network()
		self.build_network()
		# Stuff for building the big-batch and optimize training computations
		self._big_batch_feed = [{},{}]
		self._batch_count = [0,0]
		self._train_batch_size = flags.batch_size*flags.big_batch_size
		# Statistics
		self._train_statistics = Statistics(flags.episode_count_for_evaluation)
		#=======================================================================
		# self.loss_distribution_estimator = RunningMeanStd(batch_size=flags.batch_size)
		# self.actor_loss_is_too_small = False
		#=======================================================================

	def sample_actions(self, actor_batch, mean=False, std=None):
		action_batch = []
		hot_action_batch = []
		for h,actor_head in enumerate(actor_batch):
			if is_continuous_control(self.policy_heads[h]['depth']):
				new_policy_batch = tf.transpose(actor_head, [1, 0, 2])
				new_mean_batch, new_std_batch = new_policy_batch[0], new_policy_batch[1]
				distribution = Normal(new_mean_batch, new_std_batch if std is None else std*tf.ones_like(new_mean_batch))
				if not mean:
					sample_batch = distribution.sample()
					action = tf.clip_by_value(sample_batch, -1,1, name='action_clipper')
				else:
					action = distribution.mean()
				action_batch.append(action) # Sample action batch in forward direction, use old action in backward direction
				hot_action_batch.append(action)
			else: # discrete control
				distribution = Categorical(actor_head)
				if not mean:
					action = distribution.sample(one_hot=False) # Sample action batch in forward direction, use old action in backward direction
				else:
					action = distribution.mean()
				action_batch.append(action)
				hot_action_batch.append(distribution.get_sample_one_hot(action))
		# Give self esplicative name to output for easily retrieving it in frozen graph
		# tf.identity(action_batch, name="action")
		return action_batch, hot_action_batch

	def get_main_network_partitions(self):
		pass

	def build_network(self):
		pass

	def _build_train_feed(self, info_dict):
		pass

	def _train(self, feed_dict, replay=False):
		pass

	def build_fetch_maps(self):
		pass

	def get_external_network_partitions(self):
		return [['Reward']]

	def get_auxiliary_network_partitions(self):
		return []

	def get_internal_network_partitions(self):
		return self.get_main_network_partitions() + self.get_auxiliary_network_partitions()

	def get_network_partitions(self):
		return self.get_internal_network_partitions() + self.get_external_network_partitions()

	def initialize_network(self):
		self.network = {}
		# Build intrinsic reward network here because we need its internal state for building actor and critic
		self.network['Reward'] = IntrinsicReward_Network(id=self.id, scope_dict={'self': "IRNet{0}".format(self.id)}, training=self.training)
		# Build internal partitions
		for pid,partition_group in enumerate(self.get_internal_network_partitions()):
			node_id = self.id
			parent_id = self.parent.id
			sibling_id = self.sibling.id
			scope_dict = {
				'self': "Net{0}_G{1}".format(node_id,pid),
				'parent': "Net{0}_G{1}".format(parent_id,pid),
				'sibling': "Net{0}_G{1}".format(sibling_id,pid)
			}
			for p in partition_group:
				self.network[p] = eval('{}_Network'.format(flags.network_configuration))(
					id=node_id, 
					policy_heads=self.policy_heads,
					scope_dict=scope_dict, 
					training=self.training,
					value_count=self.value_count,
					state_scaler=self.state_scaler
				)
		
	def get_statistics(self):
		return self._train_statistics.get()
	
	def build_input_placeholders(self):
		print( "Building network {} input placeholders".format(self.id) )
		self.constrain_replay = flags.constraining_replay and flags.replay_mean > 0
		self.is_replayed_batch = self._scalar_placeholder(dtype=tf.bool, batch_size=1, name="replay")
		self.state_mean_batch = [self._state_placeholder(shape=head['shape'], batch_size=1, name="state_mean{}".format(i)) for i,head in enumerate(self.state_heads)] 
		self.state_std_batch = [self._state_placeholder(shape=head['shape'], batch_size=1, name="state_std{}".format(i)) for i,head in enumerate(self.state_heads)]
		self.state_batch = [self._state_placeholder(shape=head['shape'], name="state{}".format(i)) for i,head in enumerate(self.state_heads)]
		self.new_state_batch = [self._state_placeholder(shape=head['shape'], name="new_state{}".format(i)) for i,head in enumerate(self.state_heads)]
		self.size_batch = self._scalar_placeholder(dtype=tf.int32, name="size")
		self.terminal_batch = self._scalar_placeholder(dtype=tf.bool, name="terminal")
		for i,state in enumerate(self.state_batch):
			print( "	[{}]State{} shape: {}".format(self.id, i, state.get_shape()) )
		for i,state in enumerate(self.new_state_batch):
			print( "	[{}]New State{} shape: {}".format(self.id, i, state.get_shape()) )
		self.reward_batch = self._shaped_placeholder(name="reward", shape=[None,2])
		print( "	[{}]Reward shape: {}".format(self.id, self.reward_batch.get_shape()) )
		self.cumulative_return_batch = self._value_placeholder("cumulative_return")
		print( "	[{}]Cumulative Return shape: {}".format(self.id, self.cumulative_return_batch.get_shape()) )
		self.advantage_batch = self._value_placeholder("advantage")
		print( "	[{}]Advantage shape: {}".format(self.id, self.advantage_batch.get_shape()) )
		self.old_state_value_batch = self._value_placeholder("old_state_value")
		self.old_policy_batch = [self._policy_placeholder(policy_size=head['size'], policy_depth=head['depth'], name="old_policy{}".format(i)) for i,head in enumerate(self.policy_heads)]
		self.old_action_batch = [self._action_placeholder(policy_size=head['size'], policy_depth=head['depth'], name="old_action_batch{}".format(i)) for i,head in enumerate(self.policy_heads)]
		if self.has_masked_actions:
			self.old_action_mask_batch = [self._action_placeholder(policy_size=head['size'], policy_depth=1, name="old_action_mask_batch{}".format(i)) for i,head in enumerate(self.policy_heads)]
				
	def _policy_placeholder(self, policy_size, policy_depth, name=None, batch_size=None):
		if is_continuous_control(policy_depth):
			shape = [batch_size,2,policy_size]
		else: # Discrete control
			shape = [batch_size,policy_size,policy_depth] if policy_size > 1 else [batch_size,policy_depth]
		return tf.placeholder(dtype=self.parameters_type, shape=shape, name=name)
			
	def _action_placeholder(self, policy_size, policy_depth, name=None, batch_size=None):
		shape = [batch_size]
		if policy_size > 1 or is_continuous_control(policy_depth):
			shape.append(policy_size)
		if policy_depth > 1:
			shape.append(policy_depth)
		return tf.placeholder(dtype=self.parameters_type, shape=shape, name=name)

	def _shaped_placeholder(self, name=None, shape=None, dtype=None):
		if dtype is None:
			dtype=self.parameters_type
		return tf.placeholder(dtype=dtype, shape=shape, name=name)
		
	def _value_placeholder(self, name=None, batch_size=None, dtype=None):
		return self._shaped_placeholder(name=name, shape=[batch_size,self.value_count], dtype=dtype)
	
	def _scalar_placeholder(self, name=None, batch_size=None, dtype=None):
		return self._shaped_placeholder(name=name, shape=[batch_size], dtype=dtype)
		
	def _state_placeholder(self, shape, name=None, batch_size=None):
		shape = [batch_size] + list(shape)
		input = tf.zeros(shape if batch_size is not None else [1] + shape[1:], dtype=self.parameters_type) # default value
		return tf.placeholder_with_default(input=input, shape=shape, name=name) # with default we can use batch normalization directly on it
		
	def _get_internal_state(self):
		return tuple(
			self.network[p].internal_final_state 
			for p in flatten(self.get_network_partitions())
			if self.network[p].use_internal_state
		)
	
	def _get_internal_state_feed(self, internal_states):
		if not flags.network_has_internal_state:
			return {}
		feed_dict = {}
		i = 0
		for partition in flatten(self.get_network_partitions()):
			network_partition = self.network[partition]
			if network_partition.use_internal_state:
				partition_batch_states = [
					network_partition.internal_default_state if internal_state is None else internal_state[i]
					for internal_state in internal_states
				]
				for j, initial_state in enumerate(zip(*partition_batch_states)):
					feed_dict.update( {network_partition.internal_initial_state[j]: initial_state} )
				i += 1
		return feed_dict

	@staticmethod
	def _format_internal_state(new_internal_states, batch_count):
		if len(new_internal_states) == 0:
			return [new_internal_states]*batch_count
		else:
			return [
				[
					[
						sub_partition_new_internal_state[i]
						for sub_partition_new_internal_state in partition_new_internal_states
					]
					for partition_new_internal_states in new_internal_states
				]
				for i in range(batch_count)
			]

	def _get_multihead_feed(self, source, target):
		# Action and policy may have multiple heads, swap 1st and 2nd axis of source with zip*
		return { t:s for t,s in zip(target, zip(*source)) }

	def build_optimizer(self, optimization_algoritmh):
		print("Gradient {} optimized by {}".format(self.id, optimization_algoritmh))
		# global step
		global_step = tf.compat.v1.train.get_or_create_global_step()
		# learning rate
		learning_rate = tf_utils.get_annealable_variable(
			function_name=flags.alpha_annealing_function, 
			initial_value=flags.alpha, 
			global_step=global_step, 
			decay_steps=flags.alpha_decay_steps, 
			decay_rate=flags.alpha_decay_rate
		) if flags.alpha_decay else flags.alpha
		# gradient optimizer
		optimizers = [
			tf_utils.get_optimization_function(optimization_algoritmh)(learning_rate=learning_rate)
			for p_group in self.get_network_partitions()
		]
		optimizers[0].iterations = global_step
		gradient_optimizer_dict = {
			p: optimizers[i]
			for i,p_group in enumerate(self.get_network_partitions())
			for p in p_group
		}
		return gradient_optimizer_dict,global_step
	
	def get_shared_keys(self, partitions=None):
		if partitions is None:
			partitions = self.get_network_partitions()
		# set removes duplicates
		key_list = set(flatten(self.network[p].shared_keys for p in flatten(partitions)))
		return sorted(key_list, key=lambda x: x.name)
	
	def get_update_keys(self, partitions=None):
		if partitions is None:
			partitions = self.get_network_partitions()
		# set removes duplicates
		key_list = set(flatten(self.network[p].update_keys for p in flatten(partitions)))
		return sorted(key_list, key=lambda x: x.name)

	def _get_train_op(self, global_step, optimizer, loss, shared_keys, update_keys, global_keys):
		with tf.control_dependencies(update_keys): # control_dependencies is for batch normalization
			shared_keys, global_keys = zip(*(
				(k1,k2) 
				for g,k1,k2 in zip(tf.gradients(loss, shared_keys),shared_keys,global_keys) 
				if g is not None
			))
			grads = optimizer.get_gradients(loss, shared_keys)
			grads = tf.distribute.get_replica_context().all_reduce('mean', grads)
			print(f'Gradients: {len(grads)}, Global Keys: {len(global_keys)}')
			global_grads_and_vars = [(g,k) for g,k in zip(grads, global_keys) if g is not None]
			return optimizer.apply_gradients(global_grads_and_vars)
		
	def bind_sync(self, src_network, name=None):
		with tf.name_scope(name, "Sync{0}".format(self.id),[]) as name:
			src_vars = src_network.get_shared_keys()
			dst_vars = self.get_shared_keys()
			sync_ops = []
			for(src_var, dst_var) in zip(src_vars, dst_vars):
				sync_op = tf.assign(dst_var, src_var) # no need for locking dst_var
				sync_ops.append(sync_op)
			self.sync_op = tf.group(*sync_ops, name=name)
				
	def sync(self):
		tf.get_default_session().run(fetches=self.sync_op)

	def build_loss(self, global_step, partition_list):
		return {
			p: self._loss_builder[p](global_step)
			for p in partition_list
			if p in self._loss_builder
		}
		
	def setup_local_loss_minimisation(self, gradient_optimizer_dict, global_step, global_agent): # minimize loss and apply gradients to global vars.
		self.loss_dict = self.build_loss(global_step, list(gradient_optimizer_dict.keys()))
		self.build_fetch_maps()
		self.train_operations_dict = {}
		for p,optimization_fn in gradient_optimizer_dict.items():
			if p not in self.loss_dict:
				continue
			self.train_operations_dict[p] = self._get_train_op(
				global_step=global_step,
				optimizer=optimization_fn, 
				loss=sum(self.loss_dict[p]), 
				shared_keys=self.get_shared_keys([p]), 
				global_keys=global_agent.get_shared_keys([p]),
				update_keys=self.get_update_keys([p])
			)
		
	def prepare_train(self, info_dict, replay):
		''' Prepare training batch, then _train once using the biggest possible batch '''
		train_type = 1 if replay else 0
		# Get global feed
		current_global_feed = self._big_batch_feed[train_type]
		# Build local feed
		local_feed = self._build_train_feed(info_dict)
		# Merge feed dictionary
		for key,value in local_feed.items():
			if key not in current_global_feed:
				current_global_feed[key] = deque(maxlen=self._train_batch_size) # Initializing the main_feed_dict 
			current_global_feed[key].extend(value)
		# Increase the number of batches composing the big batch
		self._batch_count[train_type] += 1
		if self._batch_count[train_type]%flags.big_batch_size == 0: # can _train
			# Reset batch counter
			self._batch_count[train_type] = 0
			# Reset big-batch (especially if network_has_internal_state) otherwise when in GPU mode it's more time and memory efficient to not reset the big-batch, in order to keep its size fixed
			self._big_batch_feed[train_type] = {}
			#########################################################
			# Train
			# Add replay boolean to feed dictionary
			current_global_feed.update( {self.is_replayed_batch: [replay]} )
			# Intrinsic Reward
			if self.with_intrinsic_reward:
				state_mean, state_std = info_dict['state_mean'], info_dict['state_std']
				current_global_feed.update( self._get_multihead_feed(target=self.state_mean_batch, source=[state_mean]) )
				current_global_feed.update( self._get_multihead_feed(target=self.state_std_batch, source=[state_std]) )
			return self._train(feed_dict=current_global_feed, replay=replay)
		return None

	def fetch_info(self, fetch_label_list, info_dict):
		feed_label_set = set(list(info_dict.keys()))
		fetch_label_set = set(fetch_label_list)
		# Build minimised feed dict
		feed_dict = {}
		multihead_set = set(['states', 'new_states', 'policies', 'actions', 'hot_actions', 'state_mean', 'state_std'])
		for feed_label in feed_label_set:
			if self.feed_map.get(feed_label,None) == None:
				continue
			if feed_label in multihead_set:
				feed_dict.update(self._get_multihead_feed(target=self.feed_map[feed_label], source=info_dict[feed_label]))
			elif feed_label  == 'internal_states':
				if flags.network_has_internal_state:
					feed_dict.update( self._get_internal_state_feed( info_dict['internal_states'] ) )
					feed_dict.update( {self.feed_map['sizes']: info_dict['sizes']} )
			else:
				feed_dict.update( {self.feed_map[feed_label]: info_dict[feed_label]} )
		# Build minimised fetch dict
		fetch_dict = {
			fetch_label: self.fetch_map[fetch_label]
			for fetch_label in fetch_label_set
			if self.fetch_map.get(fetch_label,None) != None
		}
		# Get results
		result_dict = tf.get_default_session().run(
			fetches=fetch_dict, 
			feed_dict=feed_dict
		)
		# Format results
		for k,v in result_dict.items():
			if k in multihead_set: # Properly format for output: action and policy may have multiple heads, swap 1st and 2nd axis
				result_dict[k] = tuple(zip(*v))
			elif k == 'new_internal_states':
				result_dict[k] = self._format_internal_state(v, len(info_dict['sizes']))
		return result_dict
