"""
XADDPG - eXplanation-Aware Deep Deterministic Policy Gradient
==============================================

Detailed documentation:
https://docs.ray.io/en/master/rllib-algorithms.html#deep-deterministic-policy-gradients-ddpg-td3
"""  # noqa: E501

from agents.xadqn import xadqn_execution_plan
from ray.rllib.agents.ddpg.ddpg import DDPGTrainer, DEFAULT_CONFIG as DDPG_DEFAULT_CONFIG

XADDPG_DEFAULT_CONFIG = DDPG_DEFAULT_CONFIG
XADDPG_DEFAULT_CONFIG["prioritized_replay"] = True
XADDPG_DEFAULT_CONFIG["buffer_options"] = {
	'priority_id': "weights", # one of the following: gains, importance_weights, rewards, prev_rewards, action_logp
	'priority_aggregation_fn': 'lambda x: np.sum(np.abs(x))', # a reduce function (from a list of numbers to a number)
	'size': 50000, 
	'alpha': 0.6, 
	'beta': 0.4, 
	'epsilon': 1e-4, # Epsilon to add to the TD errors when updating priorities.
	'prioritized_drop_probability': 1, 
	'global_distribution_matching': False, 
	'prioritised_cluster_sampling': False, 
}
XADDPG_DEFAULT_CONFIG["clustering_scheme"] = "moving_best_extrinsic_reward_with_type" # one of the following: none, extrinsic_reward, moving_best_extrinsic_reward, moving_best_extrinsic_reward_with_type, reward_with_type
XADDPG_DEFAULT_CONFIG["batch_mode"] = "complete_episodes" # can be equal to 'truncate_episodes' only when 'clustering_scheme' is 'none'

XADDPGTrainer = DDPGTrainer.with_updates(
	name="XADDPG", 
	execution_plan=xadqn_execution_plan,
)
