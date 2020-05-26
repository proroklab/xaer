# -*- coding: utf-8 -*-
import numpy as np
from agent.algorithm.ac_algorithm import merge_splitted_advantages

class pruned_gain_estimate():
	requirement = {
		'priority_update_after_replay': True,
		'importance_weight': True,
		'advantage': True
	}

	@staticmethod
	def get(batch, agents):
		advantages, importance_weights = batch.get_all_actions(actions=['advantages','importance_weights'], agents=agents)
		merged_advantages = np.array(list(map(merge_splitted_advantages,advantages)))
		gains = merged_advantages*np.where(importance_weights > 1, importance_weights, 0)
		return np.sum(gains)

class clipped_gain_estimate():
	requirement = {
		'priority_update_after_replay': True,
		'importance_weight': True,
		'advantage': True
	}

	@staticmethod
	def get(batch, agents):
		advantages, importance_weights = batch.get_all_actions(actions=['advantages','importance_weights'], agents=agents)
		merged_advantages = np.array(list(map(merge_splitted_advantages,advantages)))
		# gains = []
		# for adv, rho in zip(extrinsic_advantages,importance_weights):
		# 	epsilon = 0.1
		# 	if 1-epsilon <= rho <= 1: # new_policy == old_policy or (action != new_policy and action != old_policy)
		# 		gains.append(rho*adv)
		# 	else:
		# 		if rho < 1-epsilon: # action == new_policy and action != old_policy
		# 			gains.append(rho*adv)
		# 		else: # action != new_policy and action == old_policy
		# 			gains.append(adv)
		gains = merged_advantages*np.minimum(1.,importance_weights)
		return np.sum(gains)

class clipped_mean_gain_estimate():
	requirement = {
		'priority_update_after_replay': True,
		'importance_weight': True,
		'advantage': True
	}

	@staticmethod
	def get(batch, agents):
		advantages, importance_weights = batch.get_all_actions(actions=['advantages','importance_weights'], agents=agents)
		merged_advantages = np.array(list(map(merge_splitted_advantages,advantages)))
		gains = merged_advantages*np.minimum(1.,importance_weights)
		return np.mean(gains)

class clipped_best_gain_estimate():
	requirement = {
		'priority_update_after_replay': True,
		'importance_weight': True,
		'advantage': True
	}

	@staticmethod
	def get(batch, agents):
		advantages, importance_weights = batch.get_all_actions(actions=['advantages','importance_weights'], agents=agents)
		merged_advantages = np.array(list(map(merge_splitted_advantages,advantages)))
		gains = merged_advantages*np.minimum(1.,importance_weights)
		return np.mean(gains)+np.std(gains)

class unclipped_gain_estimate():
	requirement = {
		'priority_update_after_replay': True,
		'importance_weight': True,
		'advantage': True
	}

	@staticmethod
	def get(batch, agents):
		advantages, importance_weights = batch.get_all_actions(actions=['advantages','importance_weights'], agents=agents)
		merged_advantages = np.array(list(map(merge_splitted_advantages,advantages)))
		gain = merged_advantages*np.array(importance_weights)
		return np.sum(gain)

class unclipped_mean_gain_estimate():
	requirement = {
		'priority_update_after_replay': True,
		'importance_weight': True,
		'advantage': True
	}

	@staticmethod
	def get(batch, agents):
		advantages, importance_weights = batch.get_all_actions(actions=['advantages','importance_weights'], agents=agents)
		merged_advantages = np.array(list(map(merge_splitted_advantages,advantages)))
		gain = merged_advantages*np.array(importance_weights)
		return np.mean(gain)

class unclipped_best_gain_estimate():
	requirement = {
		'priority_update_after_replay': True,
		'importance_weight': True,
		'advantage': True
	}

	@staticmethod
	def get(batch, agents):
		advantages, importance_weights = batch.get_all_actions(actions=['advantages','importance_weights'], agents=agents)
		merged_advantages = np.array(list(map(merge_splitted_advantages,advantages)))
		gain = merged_advantages*np.array(importance_weights)
		return np.mean(gain)+np.std(gain)

class surprise():
	requirement = {
		'priority_update_after_replay': True,
		'intrinsic_reward': True,
	}

	@staticmethod
	def get(batch, agents):
		return batch.get_cumulative_reward(agents)[-1]

class cumulative_extrinsic_return():
	requirement = {
		'priority_update_after_replay': False,
	}

	@staticmethod
	def get(batch, agents):
		return batch.get_cumulative_reward(agents)[0]

class transition_prediction_error():
	requirement = {
		'priority_update_after_replay': True,
		'transition_prediction_error': True
	}

	@staticmethod
	def get(batch, agents):
		return batch.get_cumulative_action('transition_prediction_errors', agents)

