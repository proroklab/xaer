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

from xarl.experience_buffers.replay_ops import MixInReplay, get_clustered_replay_buffer, assign_types, get_update_replayed_batch_fn, xa_make_learner_thread, add_buffer_metrics
from xarl.utils.misc import accumulate
from xarl.agents.xappo_loss.xappo_tf_loss import xappo_surrogate_loss as tf_xappo_surrogate_loss
from xarl.agents.xappo_loss.xappo_torch_loss import xappo_surrogate_loss as torch_xappo_surrogate_loss
from xarl.experience_buffers.replay_buffer import get_batch_infos, get_batch_uid

XAPPO_EXTRA_OPTIONS = {
	"_use_trajectory_view_api": False, # important
	# "lambda": .95, # GAE(lambda) parameter. Taking lambda < 1 introduces bias only when the value function is inaccurate.
	# "batch_mode": "complete_episodes", # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes', otherwise it can also be 'truncate_episodes'.
	# "vtrace": False, # Formula for computing the advantages: batch_mode==complete_episodes implies vtrace==False, thus gae==True.
	##########################################
	"rollout_fragment_length": 2**3, # Number of transitions per batch in the experience buffer
	"train_batch_size": 2**9, # Number of transitions per train-batch
	"replay_proportion": 4, # Set a p>0 to enable experience replay. Saved samples will be replayed with a p:1 proportion to new data samples.
	##########################################
	"gae_with_vtrace": False, # Useful when default "vtrace" is not active. Formula for computing the advantages: it combines GAE with V-Trace.
	"prioritized_replay": True, # Whether to replay batches with the highest priority/importance/relevance for the agent.
	"update_advantages_when_replaying": True, # Whether to recompute advantages when updating priorities.
	"learning_starts": 2**6, # How many batches to sample before learning starts. Every batch has size 'rollout_fragment_length' (default is 50).
	"buffer_options": {
		'priority_id': 'gains', # Which batch column to use for prioritisation. One of the following: gains, advantages, rewards, prev_rewards, action_logp.
		'priority_lower_limit': None, # A value lower than the lowest possible priority. It depends on the priority_id. By default in DQN and DDPG it is td_error 0, while in PPO it is gain None.
		'priority_aggregation_fn': 'np.mean', # A reduction that takes as input a list of numbers and returns a number representing a batch priority.
		'cluster_size': None, # Maximum number of batches stored in a cluster (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'rollout_fragment_length' (default is 50).
		'global_size': 2**12, # Maximum number of batches stored in all clusters (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'rollout_fragment_length' (default is 50).
		'min_cluster_size_proportion': 1, # Let X be the minimum cluster's size, and q be the min_cluster_size_proportion, then the cluster's size is guaranteed to be in [X, X+qX]. This shall help having a buffer reflecting the real distribution of tasks (where each task is associated to a cluster), thus avoiding over-estimation of task's priority.
		'prioritization_alpha': 0.5, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'prioritization_importance_beta': 0.4, # To what degree to use importance weights (0 - no corrections, 1 - full correction).
		'prioritization_importance_eta': 1e-2, # Used only if priority_lower_limit is None. A value > 0 that enables eta-weighting, thus allowing for importance weighting with priorities lower than 0 if beta is > 0. Eta is used to avoid importance weights equal to 0 when the sampled batch is the one with the highest priority. The closer eta is to 0, the closer to 0 would be the importance weight of the highest-priority batch.
		'prioritization_epsilon': 1e-6, # prioritization_epsilon to add to a priority so that it is never equal to 0.
		'prioritized_drop_probability': 0, # Probability of dropping the batch having the lowest priority in the buffer.
		'global_distribution_matching': False, # Whether to use a random number rather than the batch priority during prioritised dropping. If True then: At time t the probability of any experience being the max experience is 1/t regardless of when the sample was added, guaranteeing that (when prioritized_drop_probability==1) at any given time the sampled experiences will approximately match the distribution of all samples seen so far.
		'cluster_prioritisation_strategy': 'highest', # Whether to select which cluster to replay in a prioritised fashion -- 4 options: None; 'highest' - clusters with the highest priority are more likely to be sampled; 'average' - prioritise the cluster with priority closest to the average cluster priority; 'above_average' - prioritise the cluster with priority closest to the cluster with the smallest priority greater than the average cluster priority.
		'cluster_level_weighting': False, # Whether to use only cluster-level information to compute importance weights rather than the whole buffer.
	},
	"clustering_scheme": "multiple_types", # Which scheme to use for building clusters. One of the following: "none", "reward_against_zero", "reward_against_mean", "multiple_types_with_reward_against_mean", "multiple_types_with_reward_against_zero", "type_with_reward_against_mean", "multiple_types", "type".
	"cluster_with_episode_type": False, # Most useful with sparse-reward environments. Whether to cluster experience using information at episode-level. It requires "batch_mode" == "complete_episodes".
	"cluster_overview_size": 2, # cluster_overview_size <= train_batch_size. If None, then cluster_overview_size is automatically set to train_batch_size. -- When building a single train batch, do not sample a new cluster before x batches are sampled from it. The closer cluster_overview_size is to train_batch_size, the faster is the batch sampling procedure.
	"collect_cluster_metrics": False, # Whether to collect metrics about the experience clusters. It consumes more resources.
	"sample_also_from_buffer_of_recent_elements": False, # Whether to sample in a randomised fashion from both a non-prioritised buffer of most recent elements and the XA prioritised buffer.
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
	# Add advantages, do it after computing action_importance_ratio (used by gae-v)
	if policy.config["update_advantages_when_replaying"] or Postprocessing.ADVANTAGES not in sample_batch:
		if sample_batch[SampleBatch.DONES][-1]:
			last_r = 0.0
		# Trajectory has been truncated -> last r=VF estimate of last obs.
		else:
			# Input dict is provided to us automatically via the Model's
			# requirements. It's a single-timestep (last one in trajectory)
			# input_dict.
			if policy.config["_use_trajectory_view_api"]:
				# Create an input dict according to the Model's requirements.
				input_dict = policy.model.get_input_dict(sample_batch, index=-1)
				last_r = policy._value(**input_dict)
			# TODO: (sven) Remove once trajectory view API is all-algo default.
			else:
				next_state = []
				for i in range(policy.num_state_tensors()):
					next_state.append(sample_batch["state_out_{}".format(i)][-1])
				last_r = policy._value(sample_batch[SampleBatch.NEXT_OBS][-1],
									   sample_batch[SampleBatch.ACTIONS][-1],
									   sample_batch[SampleBatch.REWARDS][-1],
									   *next_state)

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
	sample_batch['gains'] = sample_batch['action_importance_ratio'] * sample_batch[Postprocessing.ADVANTAGES]
	if policy.config["buffer_options"]["prioritization_importance_beta"] and 'weights' not in sample_batch:
		sample_batch['weights'] = np.ones_like(sample_batch[SampleBatch.REWARDS])
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
	local_replay_buffer, clustering_scheme = get_clustered_replay_buffer(config)
	rollouts = ParallelRollouts(workers, mode="async", num_async=config["max_sample_requests_in_flight_per_worker"])
	local_worker = workers.local_worker()

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
			sample_also_from_buffer_of_recent_elements=config["sample_also_from_buffer_of_recent_elements"],
		)) \
		.flatten() \
		.combine(ConcatBatches(min_batch_size=config["train_batch_size"]))

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

	# def update_priorities(item):
	# 	samples, info_dict = item
	# 	if not config.get("prioritized_replay"):
	# 		return info_dict
	# 	# IMPORTANT: split train-batch into replay-batches, using batch_uid, before updating priorities
	# 	policy_batch_list = []
	# 	for policy_id, batch in samples.policy_batches.items():
	# 		sub_batch_indexes = [
	# 			i
	# 			for i,infos in enumerate(batch['infos'])
	# 			if "batch_uid" in infos
	# 		] + [batch.count]
	# 		sub_batch_iter = (
	# 			batch.slice(sub_batch_indexes[j], sub_batch_indexes[j+1])
	# 			for j in range(len(sub_batch_indexes)-1)
	# 		)
	# 		sub_batch_iter = unique_everseen(sub_batch_iter, key=get_batch_uid)
	# 		for i,sub_batch in enumerate(sub_batch_iter):
	# 			if i >= len(policy_batch_list):
	# 				policy_batch_list.append({})
	# 			policy_batch_list[i][policy_id] = sub_batch
	# 	for policy_batch in policy_batch_list:
	# 		local_replay_buffer.update_priorities(policy_batch)
	# 	return samples.count, info_dict
	# dequeue_op = Dequeue(learner_thread.outqueue, check=learner_thread.is_alive) \
	# 	.for_each(update_priorities) \
	# 	.for_each(record_steps_trained)
	dequeue_op = Dequeue(learner_thread.outqueue, check=learner_thread.is_alive) \
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
