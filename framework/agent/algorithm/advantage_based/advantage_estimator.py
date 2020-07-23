# -*- coding: utf-8 -*-
import numpy as np
from utils.misc import accumulate
import options
flags = options.get()

# Han, Seungyul, and Youngchul Sung. "Dimension-Wise Importance Sampling Weight Clipping for Sample-Efficient Reinforcement Learning." arXiv preprint arXiv:1905.02363 (2019).
def gae_v(gamma, last_value, reversed_reward, reversed_value, reversed_importance_weight, **args):
	def generalized_advantage_estimator_with_vtrace(gamma, lambd, last_value, reversed_reward, reversed_value, reversed_rho):
		reversed_rho = np.minimum(1.0, reversed_rho)
		def get_return(last_gae, last_value, last_rho, reward, value, rho):
			new_gae = reward + gamma*last_value - value + gamma*lambd*last_gae
			return new_gae, value, rho, last_rho*new_gae
		reversed_cumulative_advantage, _, _, _ = zip(*accumulate(
			iterable=zip(reversed_reward, reversed_value, reversed_rho), 
			func=lambda cumulative_value,reward_value_rho: get_return(
				last_gae=cumulative_value[3], 
				last_value=cumulative_value[1], 
				last_rho=cumulative_value[2], 
				reward=reward_value_rho[0], 
				value=reward_value_rho[1],
				rho=reward_value_rho[2],
			),
			initial_value=(0.,last_value,1.,0.) # initial cumulative_value
		))
		reversed_cumulative_return = tuple(map(lambda adv,val,rho: rho*adv+val, reversed_cumulative_advantage, reversed_value, reversed_rho))
		return reversed_cumulative_return, reversed_cumulative_advantage
	return generalized_advantage_estimator_with_vtrace(
		gamma=gamma, 
		lambd=flags.advantage_lambda, 
		last_value=last_value, 
		reversed_reward=reversed_reward, 
		reversed_value=reversed_value,
		reversed_rho=reversed_importance_weight
	)

# Espeholt, Lasse, et al. "Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures." arXiv preprint arXiv:1802.01561 (2018).
def vtrace(gamma, last_value, reversed_reward, reversed_value, reversed_importance_weight, **args):
	def v_trace(gamma, lambd, last_value, reversed_reward, reversed_value, reversed_rho):
		reversed_rho = np.minimum(1.0, reversed_rho)
		def get_return(last_advantage, last_value, value, rho, reward):
			new_advantage = rho*(reward + gamma*last_value - value) + lambd*gamma*rho*last_advantage
			new_vtrace = new_advantage + value
			return new_vtrace, new_advantage, value
		reversed_cumulative_return, reversed_cumulative_advantage, _ = zip(*accumulate(
			iterable=zip(reversed_value, reversed_rho, reversed_reward), 
			func=lambda cumulative_value,value_rho_reward: get_return(
				last_advantage=cumulative_value[1], 
				last_value=cumulative_value[2], 
				value=value_rho_reward[0], 
				rho=value_rho_reward[1],
				reward=value_rho_reward[2],
			),
			initial_value=(last_value,last_value, 0.) # initial cumulative_value
		))
		return reversed_cumulative_return, reversed_cumulative_advantage
	return v_trace(
		gamma=gamma, 
		lambd=flags.advantage_lambda, 
		last_value=last_value, 
		reversed_reward=reversed_reward, 
		reversed_value=reversed_value,
		reversed_rho=reversed_importance_weight,
	)

# Schulman, John, et al. "High-dimensional continuous control using generalized advantage estimation." arXiv preprint arXiv:1506.02438 (2015).
def gae(gamma, last_value, reversed_reward, reversed_value, **args):
	def generalized_advantage_estimator(gamma, lambd, last_value, reversed_reward, reversed_value):
		def get_return(last_gae, last_value, reward, value):
			new_gae = reward + gamma*last_value - value + gamma*lambd*last_gae
			return new_gae, value
		reversed_cumulative_advantage, _ = zip(*accumulate(
			iterable=zip(reversed_reward, reversed_value), 
			func=lambda cumulative_value,reward_value: get_return(
				last_gae=cumulative_value[0], 
				last_value=cumulative_value[1], 
				reward=reward_value[0], 
				value=reward_value[1]
			),
			initial_value=(0.,last_value) # initial cumulative_value
		))
		reversed_cumulative_return = tuple(map(lambda adv,val: adv+val, reversed_cumulative_advantage, reversed_value))
		return reversed_cumulative_return, reversed_cumulative_advantage
	return generalized_advantage_estimator(
		gamma=gamma, 
		lambd=flags.advantage_lambda, 
		last_value=last_value, 
		reversed_reward=reversed_reward, 
		reversed_value=reversed_value
	)

def vanilla(gamma, last_value, reversed_reward, reversed_value, **args):
	def vanilla(gamma, last_value, reversed_reward, reversed_value):
		def get_return(last_return, reward):
			return reward + gamma*last_return
		reversed_cumulative_return = tuple(accumulate(
			iterable=reversed_reward, 
			func=lambda cumulative_value,reward: get_return(last_return=cumulative_value, reward=reward),
			initial_value=last_value # initial cumulative_value
		))
		reversed_cumulative_advantage = tuple(map(lambda ret,val: ret-val, reversed_cumulative_return, reversed_value))
		return reversed_cumulative_return, reversed_cumulative_advantage
	return vanilla(
		gamma=gamma, 
		last_value=last_value, 
		reversed_reward=reversed_reward,
		reversed_value=reversed_value
	)
	