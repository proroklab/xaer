"""CQL (derived from SAC).
"""
from xarl.agents.xasac import xa_postprocess_nstep_and_prio, xadqn_execution_plan, XADQN_EXTRA_OPTIONS
from ray.rllib.agents.cql.cql import CQLTrainer, CQL_DEFAULT_CONFIG
from ray.rllib.agents.cql.cql_torch_policy import CQLTorchPolicy
from xarl.agents.xacql.xacql_torch_loss import cql_loss as torch_xacql_loss

XACQL_DEFAULT_CONFIG = CQLTrainer.merge_trainer_configs(
	CQL_DEFAULT_CONFIG, # For more details, see here: https://docs.ray.io/en/master/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn
	XADQN_EXTRA_OPTIONS,
	_allow_unknown_configs=True
)

########################
# XASAC's Policy
########################

XACQLTorchPolicy = CQLTorchPolicy.with_updates(
	name="XACQLTorchPolicy",
	postprocess_fn=xa_postprocess_nstep_and_prio,
	loss_fn=torch_xacql_loss,
)

def get_policy_class(config):
	if config["framework"] == "torch":
		return XACQLTorchPolicy

XACQLTrainer = CQLTrainer.with_updates(
	name="XACQL", 
	default_config=XACQL_DEFAULT_CONFIG,
	default_policy=XACQLTorchPolicy,
	get_policy_class=get_policy_class,
	after_init=None,
	execution_plan=xadqn_execution_plan,
)
