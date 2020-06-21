# -*- coding: utf-8 -*-
from collections import deque
import numpy as np
from agent.algorithm import *
from agent.worker.batch import CompositeBatch
from agent.worker.prioritization_scheme import *
from utils.buffer import Buffer, PseudoPrioritizedBuffer as PrioritizedBuffer
from utils.running_std import RunningMeanStd
from utils.important_information import ImportantInformation
from threading import Lock
import options
flags = options.get()

class NetworkManager(object):
	algorithm = eval('{}_Algorithm'.format(flags.algorithm))
	print('Algorithm:',flags.algorithm)
	# Experience Replay
	with_experience_replay = flags.replay_mean > 0
	print('With Experience Replay:',with_experience_replay)
	experience_prioritization_scheme = eval(flags.prioritization_scheme) if flags.prioritization_scheme and with_experience_replay else False
	print('Experience Prioritization Scheme:',experience_prioritization_scheme)
	prioritized_replay_with_update = experience_prioritization_scheme and experience_prioritization_scheme.requirement.get('priority_update_after_replay',False)
	print('Prioritized Replay With Update:',prioritized_replay_with_update)
	# Intrinsic Rewards
	prioritized_with_intrinsic_reward = experience_prioritization_scheme and experience_prioritization_scheme.requirement.get('intrinsic_reward',False)
	print('Prioritized With Intrinsic Reward:',prioritized_with_intrinsic_reward)
	with_intrinsic_reward = flags.intrinsic_reward or flags.use_learnt_environment_model_as_observation or prioritized_with_intrinsic_reward
	print('With Intrinsic Reward:', with_intrinsic_reward)
	prioritized_with_transition_predictor = experience_prioritization_scheme and experience_prioritization_scheme.requirement.get('transition_prediction_error',False)
	# Transition Prediction
	print('Prioritized With Transition Predictor:',prioritized_with_transition_predictor)
	with_transition_predictor = flags.with_transition_predictor or prioritized_with_transition_predictor
	print('With Transition Predictor:',with_transition_predictor)
	# Relation Extraction
	prioritized_with_relation_extraction = experience_prioritization_scheme and experience_prioritization_scheme.requirement.get('relation_extraction',False)
	print('Prioritized With Relation Extraction:',prioritized_with_transition_predictor)
	with_relation_extraction = prioritized_with_relation_extraction
	print('With Relation Extraction:',with_relation_extraction)
	# Importance Weight Extraction
	prioritized_with_importance_weight_extraction = experience_prioritization_scheme and experience_prioritization_scheme.requirement.get('importance_weight',False)
	print('Prioritized With Importance Weight Extraction:',prioritized_with_importance_weight_extraction)
	with_importance_weight_extraction = algorithm.extract_importance_weight or prioritized_with_importance_weight_extraction
	print('With Importance Weight Extraction:',with_importance_weight_extraction)
	if with_experience_replay:
		experience_buffer_lock = Lock() # Use a locking mechanism to access the buffer because buffers are shared among threads
		if experience_prioritization_scheme:
			experience_buffer = PrioritizedBuffer(
				size=flags.replay_buffer_size, 
				alpha=flags.prioritized_replay_alpha, 
				prioritized_drop_probability=flags.prioritized_drop_probability
			)
		else:
			experience_buffer = Buffer(size=flags.replay_buffer_size)
		ImportantInformation(experience_buffer, 'experience_buffer')
	
	def __init__(self, group_id, environment_info, global_network=None, training=True):
		self.training = training
		self.group_id = group_id
		self.set_model_size()
		self.global_network = global_network
		# Build agents
		self.model_list = self.build_agents(algorithm=self.algorithm, environment_info=environment_info)
		# Build global_step and gradient_optimizer
		if self.is_global_network():
			self.gradient_optimizer = self.build_gradient_optimizer()
		# Bind optimizer to global
		if not self.is_global_network():
			if self.training: # Prepare loss
				self.setup_local_loss_minimisation(self.global_network)
			self.bind_to_global(self.global_network)
		# Intrinsic reward
		self.can_compute_intrinsic_reward = False
		if self.with_intrinsic_reward:
			if flags.scale_intrinsic_reward:
				self.intrinsic_reward_scaler = [RunningMeanStd() for _ in range(self.model_size)]
				ImportantInformation(self.intrinsic_reward_scaler, 'intrinsic_reward_scaler{}'.format(self.group_id))
			# Reward manipulators
			self.intrinsic_reward_manipulator = eval(flags.intrinsic_reward_manipulator) if flags.intrinsic_reward else lambda x: [0]*len(x)
			self.intrinsic_reward_mini_batch_size = int(flags.batch_size*flags.intrinsic_rewards_mini_batch_fraction)
			print('[Group{}] Intrinsic rewards mini-batch size: {}'.format(self.group_id, self.intrinsic_reward_mini_batch_size))

	def get_statistics(self):
		stats = {}
		for model in self.model_list:
			stats.update(model.get_statistics())
		return stats
			
	def is_global_network(self):
		return self.global_network is None
			
	def set_model_size(self):
		self.model_size = 1
		self.agents_set = (0,)
			
	def build_agents(self, algorithm, environment_info):
		model_list = []
		agent=algorithm(
			group_id=self.group_id,
			model_id=0,
			environment_info=environment_info, 
			training=self.training,
			with_intrinsic_reward=self.with_intrinsic_reward,
			# with_transition_predictor=self.with_transition_predictor,
		)
		model_list.append(agent)
		return model_list
		
	def sync(self):
		assert not self.is_global_network(), 'Trying to sync the global network with itself'
		# Synchronize models
		for model in self.model_list:
			model.sync()
					
	def build_gradient_optimizer(self):
		return [
			m.build_optimizer(flags.optimizer)
			for m in self.model_list
		]
	
	def setup_local_loss_minimisation(self, global_network):
		for i,(local_agent,global_agent) in enumerate(zip(self.model_list, global_network.model_list)):
			gradient_optimizer_dict, global_step = global_network.gradient_optimizer[i]
			local_agent.setup_local_loss_minimisation(gradient_optimizer_dict=gradient_optimizer_dict, global_step=global_step, global_agent=global_agent)
			
	def bind_to_global(self, global_network):
		# for synching local network with global one
		for local_agent,global_agent in zip(self.model_list, global_network.model_list):
			local_agent.bind_sync(global_agent)

	def get_model(self, id=0):
		return self.model_list[id]

	def predict_action(self, states, internal_states):
		info_dict = {
			'states': states,
			'internal_states': internal_states,
			'sizes': [1 for _ in range(len(states))] # states are from different environments with different internal states
		}
		actions, hot_actions, policies, values, new_internal_states = self.get_model().predict_action(info_dict)
		agents = [0]*len(actions)
		return actions, hot_actions, policies, values, new_internal_states, agents
	
	def _update_batch(self, batch, with_value=True, with_bootstrap=True, with_intrinsic_reward=True, with_importance_weight_extraction=True, with_transition_predictor=True, with_relation_extraction=False):
		if with_importance_weight_extraction:
			self._get_importance_weight(batch)
		if with_relation_extraction:
			self._get_extracted_relations(batch)
		# if with_transition_predictor:
		# 	self._get_transition_prediction_error(batch)
		# Intrinsic Rewards
		with_intrinsic_reward = with_intrinsic_reward and self.can_compute_intrinsic_reward
		if with_intrinsic_reward:
			self._compute_intrinsic_rewards(batch)
		# Compute values and bootstrap
		if with_value:
			self._get_value(batch)
		elif with_bootstrap:
			self._bootstrap(batch)
		# Recompute discounted cumulative reward and advantage
		if with_value or with_bootstrap or with_intrinsic_reward or (with_importance_weight_extraction and self.algorithm.extract_importance_weight):
			self._compute_discounted_cumulative_reward(batch)

	def _get_value(self, batch):
		for agent_id in range(self.model_size):
			value_batch, bootstrap_value, extra_batch = self.get_model(agent_id).predict_value({
				'states': batch.states[agent_id],
				'actions': batch.actions[agent_id],
				'policies': batch.policies[agent_id],
				'internal_states': [ batch.internal_states[agent_id][0] ], # a single internal state
				'bootstrap': [ {'state':batch.new_states[agent_id][-1]} ],
				'sizes': [ len(batch.states[agent_id]) ] # playing critic on one single batch
			})
			if extra_batch is not None:
				batch.extras[agent_id] = list(extra_batch)
			batch.values[agent_id] = list(value_batch)
			batch.bootstrap[agent_id] = bootstrap_value
			assert len(batch.states[agent_id]) == len(batch.values[agent_id]), "Number of values does not match the number of states"

	def _get_importance_weight(self, batch):
		for agent_id in range(self.model_size):
			batch.importance_weights[agent_id] = self.get_model(agent_id).get_importance_weight({
				'actions': batch.actions[agent_id],
				'action_masks': batch.action_masks[agent_id],
				'policies': batch.policies[agent_id],
				'states': batch.states[agent_id],
				'internal_states': [ batch.internal_states[agent_id][0] ], # a single internal state
				'sizes': [ len(batch.states[agent_id]) ] # playing critic on one single batch
			})
			assert len(batch.states[agent_id]) == len(batch.importance_weights[agent_id]), "Number of importance_weights does not match the number of states"

	def _get_extracted_relations(self, batch):
		for agent_id in range(self.model_size):
			batch.extracted_actor_relations[agent_id], batch.extracted_critic_relations[agent_id] = self.get_model(agent_id).get_extracted_relations({
				'states': batch.states[agent_id],
			})
			# print(batch.extracted_actor_relations[agent_id])
			assert len(batch.states[agent_id]) == len(batch.extracted_actor_relations[agent_id]), "Number of extracted_actor_relations does not match the number of states"
			assert len(batch.states[agent_id]) == len(batch.extracted_critic_relations[agent_id]), "Number of extracted_critic_relations does not match the number of states"

	def _get_transition_prediction_error(self, batch):
		for agent_id in range(self.model_size):
			batch.transition_prediction_errors[agent_id] = self.get_model(agent_id).predict_transition_relevance({
				'states': batch.states[agent_id], 
				'actions': batch.actions[agent_id], 
				'new_states': batch.new_states[agent_id],
				'rewards': batch.rewards[agent_id],
			})
			assert len(batch.states[agent_id]) == len(batch.transition_prediction_errors[agent_id]), "Number of transition_prediction_errors does not match the number of states"
			
	def _bootstrap(self, batch):
		for agent_id in range(self.model_size):
			_, _, _, (bootstrap_value,), _, _ = self.predict_action(
				states=batch.new_states[agent_id][-1:],  
				internal_states=batch.new_internal_states[agent_id][-1:]
			)
			batch.bootstrap[agent_id] = bootstrap_value

	def _compute_intrinsic_rewards(self, batch):
		for agent_id in range(self.model_size):
			# Get actual rewards
			rewards = batch.rewards[agent_id]
			manipulated_rewards = batch.manipulated_rewards[agent_id]
			# Predict intrinsic rewards
			info_dict = {
				'new_states': batch.new_states[agent_id],
				'state_mean': self.state_mean,
				'state_std':self.state_std
			}
			intrinsic_rewards = self.get_model(agent_id).predict_reward(info_dict)
			# Scale intrinsic rewards
			if flags.scale_intrinsic_reward:
				scaler = self.intrinsic_reward_scaler[agent_id]
				# Build intrinsic_reward scaler
				scaler.update(intrinsic_rewards)
				# If the reward scaler is initialized, we can compute the intrinsic reward
				if not scaler.initialized:
					continue
			# Add intrinsic rewards to batch
			if self.intrinsic_reward_mini_batch_size > 1: 
				# Keep only best intrinsic rewards
				for i in range(0, len(intrinsic_rewards), self.intrinsic_reward_mini_batch_size):
					best_intrinsic_reward_index = i+np.argmax(intrinsic_rewards[i:i+self.intrinsic_reward_mini_batch_size])
					best_intrinsic_reward = intrinsic_rewards[best_intrinsic_reward_index]
					# print(i, best_intrinsic_reward_index, best_intrinsic_reward)
					if flags.scale_intrinsic_reward:
						best_intrinsic_reward = best_intrinsic_reward/scaler.std
					rewards[best_intrinsic_reward_index][1] = best_intrinsic_reward
					manipulated_rewards[best_intrinsic_reward_index][1] = self.intrinsic_reward_manipulator([best_intrinsic_reward])[0]
				# print(best_intrinsic_reward_index,best_intrinsic_reward)
			else: 
				# Keep all intrinsic rewards
				if flags.scale_intrinsic_reward:
					intrinsic_rewards = intrinsic_rewards/scaler.std
				manipulated_intrinsic_rewards = self.intrinsic_reward_manipulator(intrinsic_rewards)
				for i in range(len(intrinsic_rewards)):
					rewards[i][1] = intrinsic_rewards[i]
					manipulated_rewards[i][1] = manipulated_intrinsic_rewards[i]
					
	def _compute_discounted_cumulative_reward(self, batch):
		batch.compute_discounted_cumulative_reward(
			agents=self.agents_set, 
			gamma=flags.gamma, 
			cumulative_return_builder=self.algorithm.get_reversed_cumulative_return
		)
		
	def _train(self, batch, replay=False, start=None, end=None):
		assert self.global_network is not None, 'Cannot directly _train the global network.'
		# Train every model
		for i,model in enumerate(self.model_list):
			batch_size = len(batch.states[i])
			# Ignore empty batches
			if batch_size == 0:
				continue
			# Check whether to slice the batch
			is_valid_start = start is not None and start != 0 and start > -batch_size
			is_valid_end = end is not None and end != 0 and end < batch_size
			do_slice = is_valid_start or is_valid_end
			if do_slice:
				if not is_valid_start:
					start = None
				if not is_valid_end:
					end = None
			# Build _train dictionary
			batch_size = len(batch.states[i][start:end])
			info_dict = {
				'states':batch.states[i][start:end] if do_slice else batch.states[i],
				'new_states':batch.new_states[i][start:end] if do_slice else batch.new_states[i],
				'actions':batch.actions[i][start:end] if do_slice else batch.actions[i],
				'action_masks':batch.action_masks[i][start:end] if do_slice else batch.action_masks[i],
				'values':batch.values[i][start:end] if do_slice else batch.values[i],
				'policies':batch.policies[i][start:end] if do_slice else batch.policies[i],
				'cumulative_returns':batch.cumulative_returns[i][start:end] if do_slice else batch.cumulative_returns[i],
				'rewards':batch.rewards[i][start:end] if do_slice else batch.rewards[i],
				'internal_state':batch.internal_states[i][start] if is_valid_start else batch.internal_states[i][0],
				'state_mean':self.state_mean,
				'state_std':self.state_std,
				'terminal': [False]*(batch_size-1) + [batch.terminal if not is_valid_end else False],
				'advantages':batch.advantages[i][start:end] if do_slice else batch.advantages[i],
			}
			# Prepare _train
			train_result = model.prepare_train(info_dict=info_dict, replay=replay)
		
	def _add_to_replay_buffer(self, batch, is_best):
		# Check whether batch is empty
		if batch.is_empty(self.agents_set):
			return False
		# Build batch type
		batch_extrinsic_reward, batch_intrinsic_reward = batch.get_cumulative_reward(self.agents_set)
		#=======================================================================
		# if batch_extrinsic_reward > 0:
		# 	print("Adding new batch with reward: extrinsic {}, intrinsic {}".format(batch_extrinsic_reward, batch_intrinsic_reward))
		#=======================================================================
		type_id = '1' if batch_extrinsic_reward > 0 else '0'
		type_id += '1' if is_best else '0'
		# Populate buffer
		params_dict = {
			'batch': batch,
			'type_id': type_id
		}
		if self.experience_prioritization_scheme:
			params_dict['priority'] = self.experience_prioritization_scheme.get(batch, self.agents_set)
		with self.experience_buffer_lock:
			self.experience_buffer.put(**params_dict)
		return True

	def try_to_replay_experience(self):
		if not self.with_experience_replay:
			return
		# Check whether experience buffer has enough elements for replaying
		if not self.experience_buffer.has_atleast(flags.replay_start):
			return
		if self.prioritized_replay_with_update:
			batch_to_update = []
		# Sample n batches from experience buffer
		n = np.random.poisson(flags.replay_mean)
		for _ in range(n):
			# Sample batch
			if self.prioritized_replay_with_update:
				with self.experience_buffer_lock:
					keyed_sample = self.experience_buffer.keyed_sample()
				batch_to_update.append(keyed_sample)
				old_batch, _, _ = keyed_sample
			else:
				with self.experience_buffer_lock:
					old_batch = self.experience_buffer.sample()
			# Replay value, without keeping experience_buffer_lock the buffer update might be not consistent anymore
			self._update_batch(
				batch= old_batch, 
				with_value= flags.recompute_value_when_replaying, 
				with_bootstrap= False, 
				with_intrinsic_reward= self.with_intrinsic_reward, 
				with_importance_weight_extraction= self.with_importance_weight_extraction, 
				with_transition_predictor= self.prioritized_with_transition_predictor,
				with_relation_extraction= self.with_relation_extraction,
			)
			# Train
			self._train(replay=True, batch=old_batch)
		# Update buffer
		if self.prioritized_replay_with_update:
			for batch, bidx, btype in batch_to_update:
				new_priority = self.experience_prioritization_scheme.get(batch, self.agents_set)
				with self.experience_buffer_lock:
					self.experience_buffer.update_priority(idx=bidx, priority=new_priority, type_id=btype)
		
	def finalize_batch(self, composite_batch, global_step):	
		self.can_compute_intrinsic_reward = global_step > flags.intrinsic_reward_step
		batch = composite_batch.get()[-1]	
		# Decide whether to compute intrinsic reward
		self._update_batch(
			batch= batch, 
			with_value= False, 
			with_bootstrap= True, 
			with_intrinsic_reward= self.with_intrinsic_reward, 
			with_importance_weight_extraction= self.with_importance_weight_extraction, 
			with_transition_predictor= self.with_transition_predictor,
			with_relation_extraction= self.with_relation_extraction,
		)
		# Train
		self._train(replay=False, batch=batch)
		# Populate replay buffer
		if self.with_experience_replay:
			# Check whether to save the whole episode list into the replay buffer
			extrinsic_reward, _ = batch.get_cumulative_reward(self.agents_set)
			is_best = extrinsic_reward > 0 # Best batches = batches that lead to positive extrinsic reward
			#===================================================================
			# # Build the best known cumulative return
			# if is_best and flags.recompute_value_when_replaying:
			# 	if composite_batch.size() > 1: # No need to recompute the cumulative return if composite batch has only 1 batch
			# 		self._compute_discounted_cumulative_reward(composite_batch)
			#===================================================================
			# Add batch to experience buffer if it is a good batch or the batch has terminated
			add_composite_batch_to_buffer = is_best or (not flags.replay_only_best_batches and batch.terminal)
			if add_composite_batch_to_buffer:
				for old_batch in composite_batch.get():
					self._add_to_replay_buffer(batch=old_batch, is_best=is_best)
				# Clear composite batch
				composite_batch.clear()
