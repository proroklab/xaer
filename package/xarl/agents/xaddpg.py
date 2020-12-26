"""
XADDPG - eXplanation-Aware Deep Deterministic Policy Gradient
==============================================

Detailed documentation:
https://docs.ray.io/en/master/rllib-algorithms.html#deep-deterministic-policy-gradients-ddpg-td3
"""  # noqa: E501

from xarl.agents.xadqn import xadqn_execution_plan, XADQN_EXTRA_OPTIONS
from ray.rllib.agents.ddpg.ddpg import DDPGTrainer, DEFAULT_CONFIG as DDPG_DEFAULT_CONFIG

XADDPG_DEFAULT_CONFIG = DDPGTrainer.merge_trainer_configs(
	DDPG_DEFAULT_CONFIG, # For more details, see here: https://docs.ray.io/en/master/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn
	XADQN_EXTRA_OPTIONS,
	_allow_unknown_configs=True
)

########################
# XADDPG's Execution Plan
########################

XADDPGTrainer = DDPGTrainer.with_updates(
	name="XADDPG", 
	default_config=XADDPG_DEFAULT_CONFIG,
	execution_plan=xadqn_execution_plan,
)
