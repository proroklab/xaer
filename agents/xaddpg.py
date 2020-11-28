"""
XADDPG - eXplanation-Aware Deep Deterministic Policy Gradient
==============================================

Detailed documentation:
https://docs.ray.io/en/master/rllib-algorithms.html#deep-deterministic-policy-gradients-ddpg-td3
"""  # noqa: E501

from agents.xadqn import xadqn_execution_plan
from ray.rllib.agents.ddpg.ddpg import DDPGTrainer, DEFAULT_CONFIG as DDPG_DEFAULT_CONFIG

XADDPG_DEFAULT_CONFIG = DDPG_DEFAULT_CONFIG
XADDPG_DEFAULT_CONFIG["buffer_options"] = {
    'size': 2**9, 
    'alpha': 0.5, 
    'prioritized_drop_probability': 0.5, 
    'global_distribution_matching': False, 
    'prioritised_cluster_sampling': True,
}
XADDPG_DEFAULT_CONFIG["worker_side_prioritization"] = True
XADDPG_DEFAULT_CONFIG["clustering_scheme"] = "moving_best_extrinsic_reward_with_type" # one of the following: none, extrinsic_reward, moving_best_extrinsic_reward, moving_best_extrinsic_reward_with_type, reward_with_type
XADDPG_DEFAULT_CONFIG["batch_mode"] = "complete_episodes"
XADDPG_DEFAULT_CONFIG["priority_weight"] = "weights" # one of the following: weigths, rewards, prev_rewards, action_logp
XADDPG_DEFAULT_CONFIG["priority_weights_aggregator"] = 'np.mean' # a reduce function (from a list of numbers to a number)

XADDPGTrainer = DDPGTrainer.with_updates(
    name="XADDPG", 
    execution_plan=xadqn_execution_plan,
)
