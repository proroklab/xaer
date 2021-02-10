"""
XADDPG - eXplanation-Aware Deep Deterministic Policy Gradient
==============================================

Detailed documentation:
https://docs.ray.io/en/master/rllib-algorithms.html#deep-deterministic-policy-gradients-ddpg-td3
"""  # noqa: E501

from xarl.agents.xadqn import xa_postprocess_nstep_and_prio, xadqn_execution_plan, XADQN_EXTRA_OPTIONS
from ray.rllib.agents.ddpg.ddpg import DDPGTrainer, DEFAULT_CONFIG as DDPG_DEFAULT_CONFIG
from ray.rllib.agents.ddpg.td3 import TD3Trainer, TD3_DEFAULT_CONFIG
from ray.rllib.agents.ddpg.ddpg_tf_policy import DDPGTFPolicy
from ray.rllib.agents.ddpg.ddpg_torch_policy import DDPGTorchPolicy

MISSING_OPTIONS = {
	"dueling": True,
	"double_q": True,
	"num_atoms": 21,
	"v_max": 10,
	"v_min": -10,
}

XADDPG_DEFAULT_CONFIG = DDPGTrainer.merge_trainer_configs(
	DDPG_DEFAULT_CONFIG, # For more details, see here: https://docs.ray.io/en/master/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn
	XADQN_EXTRA_OPTIONS,
	_allow_unknown_configs=True
)
XADDPG_DEFAULT_CONFIG = DDPGTrainer.merge_trainer_configs(
	XADDPG_DEFAULT_CONFIG,
	MISSING_OPTIONS,
	_allow_unknown_configs=True
)

XATD3_DEFAULT_CONFIG = TD3Trainer.merge_trainer_configs(
	TD3_DEFAULT_CONFIG, # For more details, see here: https://docs.ray.io/en/master/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn
	XADQN_EXTRA_OPTIONS,
	_allow_unknown_configs=True
)
XATD3_DEFAULT_CONFIG = TD3Trainer.merge_trainer_configs(
	XATD3_DEFAULT_CONFIG,
	MISSING_OPTIONS,
	_allow_unknown_configs=True
)

########################
# XADDPG's Policy
########################

XADDPGTFPolicy = DDPGTFPolicy.with_updates(
	name="XADDPGTFPolicy",
	postprocess_fn=xa_postprocess_nstep_and_prio,
)
XADDPGTorchPolicy = DDPGTorchPolicy.with_updates(
	name="XADDPGTorchPolicy",
	postprocess_fn=xa_postprocess_nstep_and_prio,
)

def get_policy_class(config):
	if config["framework"] == "torch": return XADDPGTorchPolicy
	return XADDPGTFPolicy

########################
# XADDPG's Execution Plan
########################

XADDPGTrainer = DDPGTrainer.with_updates(
	name="XADDPG", 
	default_config=XADDPG_DEFAULT_CONFIG,
	execution_plan=xadqn_execution_plan,
	get_policy_class=get_policy_class,
)

XATD3Trainer = TD3Trainer.with_updates(
    name="XATD3",
    default_config=XATD3_DEFAULT_CONFIG,
    execution_plan=xadqn_execution_plan,
	get_policy_class=get_policy_class,
)
