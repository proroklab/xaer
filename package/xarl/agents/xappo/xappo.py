"""
XAPPO - eXplanation-Aware Asynchronous Proximal Policy Optimization
==============================================

Detailed documentation:
https://docs.ray.io/en/master/rllib-algorithms.html#asynchronous-proximal-policy-optimization-appo
"""  # noqa: E501

from more_itertools import unique_everseen

from ray.rllib.agents.impala.impala import *
from ray.rllib.agents.ppo.appo import *
from ray.rllib.agents.ppo.appo_tf_policy import *
from ray.rllib.agents.ppo.ppo_tf_policy import vf_preds_fetches
from ray.rllib.agents.ppo.appo_torch_policy import AsyncPPOTorchPolicy
# from ray.rllib.evaluation.postprocessing import discount_cumsum
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch, DEFAULT_POLICY_ID
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.policy.view_requirement import ViewRequirement

from xarl.experience_buffers.replay_ops import MixInReplay, get_clustered_replay_buffer, assign_types, get_update_replayed_batch_fn, xa_make_learner_thread, add_buffer_metrics
from xarl.utils.misc import accumulate
from xarl.agents.xappo.xappo_tf_loss import xappo_surrogate_loss as tf_xappo_surrogate_loss
from xarl.agents.xappo.xappo_torch_loss import xappo_surrogate_loss as torch_xappo_surrogate_loss
from xarl.experience_buffers.replay_buffer import get_batch_infos, get_batch_uid
import random
import numpy as np

XAPPO_EXTRA_OPTIONS = {
	# "lambda": .95, # GAE(lambda) parameter. Taking lambda < 1 introduces bias only when the value function is inaccurate.
	# "batch_mode": "complete_episodes", # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes', otherwise it can also be 'truncate_episodes'.
	# "vtrace": False, # Formula for computing the advantages: batch_mode==complete_episodes implies vtrace==False, thus gae==True.
	##########################################
	"rollout_fragment_length": 2**3, # Number of transitions per batch in the experience buffer
	"train_batch_size": 2**9, # Number of transitions per train-batch
	"replay_proportion": 4, # Set a p>0 to enable experience replay. Saved samples will be replayed with a p:1 proportion to new data samples.
	"gae_with_vtrace": False, # Useful when default "vtrace" is not active. Formula for computing the advantages: it combines GAE with V-Trace.
	"prioritized_replay": True, # Whether to replay batches with the highest priority/importance/relevance for the agent.
	"update_advantages_when_replaying": True, # Whether to recompute advantages when updating priorities.
	"learning_starts": 1, # How many batches to sample before learning starts. Every batch has size 'rollout_fragment_length' (default is 50).
	##########################################
	"buffer_options": {
		'priority_id': 'gains', # Which batch column to use for prioritisation. One of the following: gains, advantages, rewards, prev_rewards, action_logp.
		'priority_lower_limit': None, # A value lower than the lowest possible priority. It depends on the priority_id. By default in DQN and DDPG it is td_error 0, while in PPO it is gain None.
		'priority_aggregation_fn': 'np.mean', # A reduction that takes as input a list of numbers and returns a number representing a batch priority.
		'cluster_size': None, # Default None, implying being equal to global_size. Maximum number of batches stored in a cluster (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'replay_sequence_length' (default is 1).
		'global_size': 2**12, # Default 50000. Maximum number of batches stored in all clusters (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'replay_sequence_length' (default is 1).
		'prioritization_alpha': 0.6, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'prioritization_importance_beta': 0, # To what degree to use importance weights (0 - no corrections, 1 - full correction).
		'prioritization_importance_eta': 1e-2, # Used only if priority_lower_limit is None. A value > 0 that enables eta-weighting, thus allowing for importance weighting with priorities lower than 0 if beta is > 0. Eta is used to avoid importance weights equal to 0 when the sampled batch is the one with the highest priority. The closer eta is to 0, the closer to 0 would be the importance weight of the highest-priority batch.
		'prioritization_epsilon': 1e-6, # prioritization_epsilon to add to a priority so that it is never equal to 0.
		'prioritized_drop_probability': 0, # Probability of dropping the batch having the lowest priority in the buffer instead of the one having the lowest timestamp. In DQN default is 0.
		'global_distribution_matching': False, # Whether to use a random number rather than the batch priority during prioritised dropping. If True then: At time t the probability of any experience being the max experience is 1/t regardless of when the sample was added, guaranteeing that (when prioritized_drop_probability==1) at any given time the sampled experiences will approximately match the distribution of all samples seen so far.
		'cluster_prioritisation_strategy': 'sum', # Whether to select which cluster to replay in a prioritised fashion -- Options: None; 'sum', 'avg', 'weighted_avg'.
		'cluster_prioritization_alpha': 1, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'cluster_level_weighting': False, # Whether to use only cluster-level information to compute importance weights rather than the whole buffer.
		'clustering_xi': 4, # Let X be the minimum cluster's size, and C be the number of clusters, and q be clustering_xi, then the cluster's size is guaranteed to be in [X, X+(q-1)CX], with q >= 1, when all clusters have reached the minimum capacity X. This shall help having a buffer reflecting the real distribution of tasks (where each task is associated to a cluster), thus avoiding over-estimation of task's priority.
		# 'clip_cluster_priority_by_max_capacity': False, # Default is False. Whether to clip the clusters priority so that the 'cluster_prioritisation_strategy' will not consider more elements than the maximum cluster capacity. In fact, until al the clusters have reached the minimum size, some clusters may have more elements than the maximum size, to avoid shrinking the buffer capacity with clusters having not enough transitions (i.e. 1 transition).
		'max_age_window': None, # Consider only batches with a relative age within this age window, the younger is a batch the higher will be its importance. Set to None for no age weighting. # Idea from: Fedus, William, et al. "Revisiting fundamentals of experience replay." International Conference on Machine Learning. PMLR, 2020.
	},
	"clustering_scheme": "HW", # Which scheme to use for building clusters. One of the following: "none", "positive_H", "H", "HW", "long_HW", "W", "long_W".
	"clustering_scheme_options": {
		"episode_window_size": 2**6, 
		"batch_window_size": 2**8, 
		"n_clusters": 8,
	},
	"cluster_selection_policy": "min", # Which policy to follow when clustering_scheme is not "none" and multiple explanatory labels are associated to a batch. One of the following: 'random_uniform_after_filling', 'random_uniform', 'random_max', 'max', 'min', 'none'
	"cluster_with_episode_type": False, # Useful with sparse-reward environments. Whether to cluster experience using information at episode-level.
	"cluster_overview_size": 1, # cluster_overview_size <= train_batch_size. If None, then cluster_overview_size is automatically set to train_batch_size. -- When building a single train batch, do not sample a new cluster before x batches are sampled from it. The closer cluster_overview_size is to train_batch_size, the faster is the batch sampling procedure.
	"collect_cluster_metrics": False, # Whether to collect metrics about the experience clusters. It consumes more resources.
	"ratio_of_samples_from_unclustered_buffer": 0, # 0 for no, 1 for full. Whether to sample in a randomised fashion from both a non-prioritised buffer of most recent elements and the XA prioritised buffer.
}
# The combination of update_insertion_time_when_sampling==True and prioritized_drop_probability==0 helps mantaining in the buffer only those batches with the most up-to-date priorities.
XAPPO_DEFAULT_CONFIG = APPOTrainer.merge_trainer_configs(
	DEFAULT_CONFIG, # For more details, see here: https://docs.ray.io/en/master/rllib-algorithms.html#asynchronous-proximal-policy-optimization-appo
	XAPPO_EXTRA_OPTIONS,
	_allow_unknown_configs=True
)

########################
# XAPPO's Policy
########################
# Han, Seungyul, and Youngchul Sung. "Dimension-Wise Importance Sampling Weight Clipping for Sample-Efficient Reinforcement Learning." arXiv preprint arXiv:1905.02363 (2019).
def gae_v(gamma, lambda_, last_value, reversed_reward, reversed_value, reversed_importance_weight):
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
		lambd=lambda_, 
		last_value=last_value, 
		reversed_reward=reversed_reward, 
		reversed_value=reversed_value,
		reversed_rho=reversed_importance_weight
	)

def compute_gae_v_advantages(rollout: SampleBatch, last_r: float, gamma: float = 0.9, lambda_: float = 1.0):
	rollout_size = len(rollout[SampleBatch.ACTIONS])
	assert SampleBatch.VF_PREDS in rollout, "values not found"
	reversed_cumulative_return, reversed_cumulative_advantage = gae_v(
		gamma, 
		lambda_, 
		last_r, 
		rollout[SampleBatch.REWARDS][::-1], 
		rollout[SampleBatch.VF_PREDS][::-1], 
		rollout["action_importance_ratio"][::-1]
	)
	rollout[Postprocessing.ADVANTAGES] = np.array(reversed_cumulative_advantage, dtype=np.float32)[::-1]
	rollout[Postprocessing.VALUE_TARGETS] = np.array(reversed_cumulative_return, dtype=np.float32)[::-1]
	assert all(val.shape[0] == rollout_size for key, val in rollout.items()), "Rollout stacked incorrectly!"
	return rollout

# TODO: (sven) Experimental method.
def get_single_step_input_dict(self, view_requirements, index="last"):
	"""Creates single ts SampleBatch at given index from `self`.

	For usage as input-dict for model calls.

	Args:
		sample_batch (SampleBatch): A single-trajectory SampleBatch object
			to generate the compute_actions input dict from.
		index (Union[int, str]): An integer index value indicating the
			position in the trajectory for which to generate the
			compute_actions input dict. Set to "last" to generate the dict
			at the very end of the trajectory (e.g. for value estimation).
			Note that "last" is different from -1, as "last" will use the
			final NEXT_OBS as observation input.

	Returns:
		SampleBatch: The (single-timestep) input dict for ModelV2 calls.
	"""
	last_mappings = {
		SampleBatch.OBS: SampleBatch.NEXT_OBS,
		SampleBatch.PREV_ACTIONS: SampleBatch.ACTIONS,
		SampleBatch.PREV_REWARDS: SampleBatch.REWARDS,
	}

	input_dict = {}
	for view_col, view_req in view_requirements.items():
		# Create batches of size 1 (single-agent input-dict).
		data_col = view_req.data_col or view_col
		if index == "last":
			data_col = last_mappings.get(data_col, data_col)
			# Range needed.
			if view_req.shift_from is not None:
				data = self[view_col][-1]
				traj_len = len(self[data_col])
				missing_at_end = traj_len % view_req.batch_repeat_value
				obs_shift = -1 if data_col in [
					SampleBatch.OBS, SampleBatch.NEXT_OBS
				] else 0
				from_ = view_req.shift_from + obs_shift
				to_ = view_req.shift_to + obs_shift + 1
				if to_ == 0:
					to_ = None
				input_dict[view_col] = np.array([
					np.concatenate(
						[data,
						 self[data_col][-missing_at_end:]])[from_:to_]
				])
			# Single index.
			else:
				data = self[data_col][-1]
				input_dict[view_col] = np.array([data])
		else:
			# Index range.
			if isinstance(index, tuple):
				data = self[data_col][index[0]:index[1] +
									  1 if index[1] != -1 else None]
				input_dict[view_col] = np.array([data])
			# Single index.
			else:
				input_dict[view_col] = self[data_col][
					index:index + 1 if index != -1 else None]

	return SampleBatch(input_dict, seq_lens=np.array([1], dtype=np.int32))

def xappo_postprocess_trajectory(policy, sample_batch, other_agent_batches=None, episode=None):
	# Add PPO's importance weights
	action_logp = policy.compute_log_likelihoods(
		actions=sample_batch[SampleBatch.ACTIONS],
		obs_batch=sample_batch[SampleBatch.CUR_OBS],
		state_batches=None, # missing, needed for RNN-based models
		prev_action_batch=None,
		prev_reward_batch=None,
	)
	old_action_logp = sample_batch[SampleBatch.ACTION_LOGP]
	sample_batch["action_importance_ratio"] = np.exp(action_logp - old_action_logp)
	if policy.config["buffer_options"]["prioritization_importance_beta"] and 'weights' not in sample_batch:
		sample_batch['weights'] = np.ones_like(sample_batch[SampleBatch.REWARDS])
	# sample_batch[Postprocessing.VALUE_TARGETS] = sample_batch[Postprocessing.ADVANTAGES] = np.ones_like(sample_batch[SampleBatch.REWARDS])
	# Add advantages, do it after computing action_importance_ratio (used by gae-v)
	if policy.config["update_advantages_when_replaying"] or Postprocessing.ADVANTAGES not in sample_batch:
		if sample_batch[SampleBatch.DONES][-1]:
			last_r = 0.0
		# Trajectory has been truncated -> last r=VF estimate of last obs.
		else:
			# Input dict is provided to us automatically via the Model's
			# requirements. It's a single-timestep (last one in trajectory)
			# input_dict.
			# Create an input dict according to the Model's requirements.
			input_dict = get_single_step_input_dict(sample_batch, policy.model.view_requirements, index="last")
			last_r = policy._value(**input_dict)

		# Adds the policy logits, VF preds, and advantages to the batch,
		# using GAE ("generalized advantage estimation") or not.
		
		if not policy.config["vtrace"] and policy.config["gae_with_vtrace"]:
			sample_batch = compute_gae_v_advantages(
				sample_batch, 
				last_r, 
				policy.config["gamma"], 
				policy.config["lambda"]
			)
		else:
			sample_batch = compute_advantages(
				sample_batch,
				last_r,
				policy.config["gamma"],
				policy.config["lambda"],
				use_gae=policy.config["use_gae"],
				use_critic=policy.config.get("use_critic", True)
			)
	# Add gains
	sample_batch['gains'] = sample_batch['action_importance_ratio']*sample_batch[Postprocessing.ADVANTAGES]
	return sample_batch

XAPPOTFPolicy = AsyncPPOTFPolicy.with_updates(
	name="XAPPOTFPolicy",
	extra_action_fetches_fn=vf_preds_fetches,
	postprocess_fn=xappo_postprocess_trajectory,
	loss_fn=tf_xappo_surrogate_loss,
)
XAPPOTorchPolicy = AsyncPPOTorchPolicy.with_updates(
	name="XAPPOTorchPolicy",
	extra_action_out_fn=vf_preds_fetches,
	postprocess_fn=xappo_postprocess_trajectory,
	loss_fn=torch_xappo_surrogate_loss,
)

def xappo_get_policy_class(config):
	if config["framework"] == "torch":
		return XAPPOTorchPolicy
	return XAPPOTFPolicy

########################
# XAPPO's Execution Plan
########################

def xappo_execution_plan(workers, config):
	random.seed(config["seed"])
	np.random.seed(config["seed"])
	local_replay_buffer, clustering_scheme = get_clustered_replay_buffer(config)
	rollouts = ParallelRollouts(workers, mode="async", num_async=config["max_sample_requests_in_flight_per_worker"])
	local_worker = workers.local_worker()
	
	def add_view_requirements(w):
		for policy in w.policy_map.values():
			policy.view_requirements[SampleBatch.INFOS] = ViewRequirement(SampleBatch.INFOS, shift=0)
			policy.view_requirements[SampleBatch.ACTION_LOGP] = ViewRequirement(SampleBatch.ACTION_LOGP, shift=0)
			policy.view_requirements[SampleBatch.NEXT_OBS] = ViewRequirement(SampleBatch.OBS, shift=1)
			policy.view_requirements[SampleBatch.VF_PREDS] = ViewRequirement(SampleBatch.VF_PREDS, shift=0)
			policy.view_requirements[Postprocessing.ADVANTAGES] = ViewRequirement(Postprocessing.ADVANTAGES, shift=0)
			policy.view_requirements[Postprocessing.VALUE_TARGETS] = ViewRequirement(Postprocessing.VALUE_TARGETS, shift=0)
			policy.view_requirements["action_importance_ratio"] = ViewRequirement("action_importance_ratio", shift=0)
			policy.view_requirements["gains"] = ViewRequirement("gains", shift=0)
			if policy.config["buffer_options"]["prioritization_importance_beta"]:
				policy.view_requirements["weights"] = ViewRequirement("weights", shift=0)
	workers.foreach_worker(add_view_requirements)
	
	# Augment with replay and concat to desired train batch size.
	train_batches = rollouts \
		.for_each(lambda batch: batch.decompress_if_needed()) \
		.for_each(lambda batch: assign_types(batch, clustering_scheme, config["rollout_fragment_length"], with_episode_type=config["cluster_with_episode_type"])) \
		.flatten() \
		.for_each(MixInReplay(
			local_buffer=local_replay_buffer,
			replay_proportion=config["replay_proportion"],
			cluster_overview_size=config["cluster_overview_size"],
			# update_replayed_fn=get_update_replayed_batch_fn(local_replay_buffer, local_worker, xappo_postprocess_trajectory) if not config['vtrace'] else lambda x:x,
			update_replayed_fn=get_update_replayed_batch_fn(local_replay_buffer, local_worker, xappo_postprocess_trajectory),
			seed=config["seed"],
		)) \
		.flatten() \
		.combine(
			ConcatBatches(
				min_batch_size=config["train_batch_size"],
				count_steps_by=config["multiagent"]["count_steps_by"],
			)
		)

	# Start the learner thread.
	# learner_thread = xa_make_learner_thread(local_worker, config)
	learner_thread = make_learner_thread(local_worker, config)
	learner_thread.start()

	# This sub-flow sends experiences to the learner.
	enqueue_op = train_batches.for_each(Enqueue(learner_thread.inqueue)) 
	# Only need to update workers if there are remote workers.
	if workers.remote_workers():
		enqueue_op = enqueue_op.zip_with_source_actor() \
			.for_each(BroadcastUpdateLearnerWeights(learner_thread, workers, broadcast_interval=config["broadcast_interval"]))

	def increase_train_steps(x):
		local_replay_buffer.increase_train_steps()
		return x
	dequeue_op = Dequeue(learner_thread.outqueue, check=learner_thread.is_alive) \
		.for_each(increase_train_steps) \
		.for_each(record_steps_trained)

	merged_op = Concurrently([enqueue_op, dequeue_op], mode="async", output_indexes=[1])

	# Callback for APPO to use to update KL, target network periodically.
	# The input to the callback is the learner fetches dict.
	if config["after_train_step"]:
		merged_op = merged_op \
			.for_each(lambda t: t[1]) \
			.for_each(config["after_train_step"](workers, config))

	standard_metrics_reporting = StandardMetricsReporting(merged_op, workers, config).for_each(learner_thread.add_learner_metrics)
	if config['collect_cluster_metrics']:
		standard_metrics_reporting = standard_metrics_reporting.for_each(lambda x: add_buffer_metrics(x,local_replay_buffer))
	return standard_metrics_reporting

XAPPOTrainer = APPOTrainer.with_updates(
	name="XAPPO", 
	default_config=XAPPO_DEFAULT_CONFIG,
	default_policy=XAPPOTFPolicy,
	get_policy_class=xappo_get_policy_class,
	execution_plan=xappo_execution_plan,
)
