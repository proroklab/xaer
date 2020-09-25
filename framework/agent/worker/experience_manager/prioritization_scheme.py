# -*- coding: utf-8 -*-
import numpy as np
from agent.algorithm.advantage_based.ac_algorithm import merge_splitted_advantages

class unclipped_gain_estimate():
	def __init__(self, algorithm):
		self.requirement = {
			'priority_update_after_replay': True,
			'importance_weight': algorithm.has_importance_weight,
			'advantage': algorithm.has_advantage,
			'td_error': algorithm.has_td_error,
		}
		self.aggregation_fn = np.sum

	def get_weighted_advantage(self, batch, agents):
		advantages, importance_weights = batch.get_all_actions(actions=['advantages','importance_weights'], agents=agents)
		merged_advantages = np.array(list(map(merge_splitted_advantages,advantages)))
		gain = merged_advantages*np.array(importance_weights)
		return self.aggregation_fn(gain)

	def get_advantage(self, batch, agents):
		(advantages,) = batch.get_all_actions(actions=['advantages'], agents=agents)
		merged_advantages = np.array(list(map(merge_splitted_advantages,advantages)))
		return self.aggregation_fn(merged_advantages)

	def get_td_error(self, batch, agents):
		(errors,) = batch.get_all_actions(actions=['td_errors'], agents=agents)
		merged_errors = np.array(list(map(merge_splitted_advantages,errors)))
		return self.aggregation_fn(merged_errors)

	def get_value(self, batch, agents):
		(values,) = batch.get_all_actions(actions=['values'], agents=agents)
		merged_values = np.array(list(map(merge_splitted_advantages,values)))
		return self.aggregation_fn(merged_values)

	def get(self, batch, agents):
		if self.requirement['importance_weight'] and self.requirement['advantage']:
			return self.get_weighted_advantage(batch, agents)
		if self.requirement['advantage']:
			return self.get_advantage(batch, agents)
		if self.requirement['td_error']:
			return self.get_td_error(batch, agents)
		return self.get_value(batch, agents)

class pruned_gain_estimate(unclipped_gain_estimate):
	def get_weighted_advantage(self, batch, agents):
		advantages, importance_weights = batch.get_all_actions(actions=['advantages','importance_weights'], agents=agents)
		merged_advantages = np.array(list(map(merge_splitted_advantages,advantages)))
		gains = merged_advantages*np.where(importance_weights > 1, importance_weights, 0)
		return np.sum(gains)

class clipped_gain_estimate(unclipped_gain_estimate):
	def get_weighted_advantage(self, batch, agents):
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

class clipped_mean_gain_estimate(clipped_gain_estimate):
	def __init__(self, algorithm):
		self.requirement = {
			'priority_update_after_replay': True,
			'importance_weight': algorithm.has_importance_weight,
			'advantage': algorithm.has_advantage,
			# 'td_error': algorithm.has_td_error,
			'td_error': algorithm.has_td_error,
		}
		self.aggregation_fn = np.mean

class clipped_best_gain_estimate(clipped_gain_estimate):
	def __init__(self, algorithm):
		self.requirement = {
			'priority_update_after_replay': True,
			'importance_weight': algorithm.has_importance_weight,
			'advantage': algorithm.has_advantage,
			# 'td_error': algorithm.has_td_error,
			'td_error': algorithm.has_td_error,
		}
		self.aggregation_fn = lambda x: np.mean(x)+np.std(x)

class unclipped_mean_gain_estimate(unclipped_gain_estimate):
	def __init__(self, algorithm):
		self.requirement = {
			'priority_update_after_replay': True,
			'importance_weight': algorithm.has_importance_weight,
			'advantage': algorithm.has_advantage,
			# 'td_error': algorithm.has_td_error,
			'td_error': algorithm.has_td_error,
		}
		self.aggregation_fn = np.mean

class unclipped_best_gain_estimate(unclipped_gain_estimate):
	def __init__(self, algorithm):
		self.requirement = {
			'priority_update_after_replay': True,
			'importance_weight': algorithm.has_importance_weight,
			'advantage': algorithm.has_advantage,
			# 'td_error': algorithm.has_td_error,
			'td_error': algorithm.has_td_error,
		}
		self.aggregation_fn = lambda x: np.mean(x)+np.std(x)

class surprise():
	def __init__(self, algorithm):
		self.requirement = {
			'priority_update_after_replay': True,
			'intrinsic_reward': True,
		}

	def get(self, batch, agents):
		return batch.get_cumulative_reward(agents)[-1]

class cumulative_extrinsic_return():
	def __init__(self, algorithm):
		self.requirement = {
			'priority_update_after_replay': False,
		}

	def get(self, batch, agents):
		return batch.get_cumulative_reward(agents)[0]
