"""
XASAC - eXplanation-Aware Soft Actor-Critic
==============================================

Detailed documentation:
https://docs.ray.io/en/master/rllib-algorithms.html#deep-deterministic-policy-gradients-ddpg-td3
"""  # noqa: E501

from xarl.agents.xadqn import xa_postprocess_nstep_and_prio, xadqn_execution_plan, XADQN_EXTRA_OPTIONS
from ray.rllib.agents.sac.sac import SACTrainer, DEFAULT_CONFIG as SAC_DEFAULT_CONFIG
from ray.rllib.agents.sac.sac_torch_policy import SACTorchPolicy
from ray.rllib.agents.sac.sac_tf_policy import SACTFPolicy
from xarl.agents.xasac_loss.xasac_tf_loss import xasac_actor_critic_loss as tf_xasac_actor_critic_loss
from xarl.agents.xasac_loss.xasac_torch_loss import xasac_actor_critic_loss as torch_xasac_actor_critic_loss

XASAC_EXTRA_OPTIONS = {**XADQN_EXTRA_OPTIONS,'tau':1e-4}
XASAC_DEFAULT_CONFIG = SACTrainer.merge_trainer_configs(
	SAC_DEFAULT_CONFIG, # For more details, see here: https://docs.ray.io/en/master/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn
	XASAC_EXTRA_OPTIONS,
	_allow_unknown_configs=True
)

########################
# XASAC's Policy
########################

XASACTFPolicy = SACTFPolicy.with_updates(
	name="XASACTFPolicy",
	postprocess_fn=xa_postprocess_nstep_and_prio,
	loss_fn=tf_xasac_actor_critic_loss,
)
XASACTorchPolicy = SACTorchPolicy.with_updates(
	name="XASACTorchPolicy",
	postprocess_fn=xa_postprocess_nstep_and_prio,
	loss_fn=torch_xasac_actor_critic_loss,
)

########################
# XADDPG's Execution Plan
########################

XASACTrainer = SACTrainer.with_updates(
	name="XASAC", 
	default_config=XASAC_DEFAULT_CONFIG,
	execution_plan=xadqn_execution_plan,
	get_policy_class=lambda config: XASACTorchPolicy if config["framework"] == "torch" else XASACTFPolicy,
)
