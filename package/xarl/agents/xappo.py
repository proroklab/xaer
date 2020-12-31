"""
XAPPO - eXplanation-Aware Asynchronous Proximal Policy Optimization
==============================================

Detailed documentation:
https://docs.ray.io/en/master/rllib-algorithms.html#asynchronous-proximal-policy-optimization-appo
"""  # noqa: E501

import collections

from ray.rllib.agents.impala.impala import *
from ray.rllib.agents.ppo.appo import *
from ray.rllib.agents.ppo.appo_tf_policy import *
from ray.rllib.agents.ppo.ppo_tf_policy import vf_preds_fetches
from ray.rllib.agents.ppo.appo_torch_policy import AsyncPPOTorchPolicy
# from ray.rllib.evaluation.postprocessing import discount_cumsum
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch, DEFAULT_POLICY_ID

from xarl.experience_buffers.replay_ops import MixInReplay, get_clustered_replay_buffer, assign_types
from xarl.utils.misc import accumulate

XAPPO_EXTRA_OPTIONS = {
	"batch_mode": "complete_episodes", # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes', otherwise it can also be 'truncate_episodes'.
	"vtrace": False, # Formula for computing the advantages: batch_mode==complete_episodes implies vtrace==False, thus gae==True.
	"replay_proportion": 2, # Set a p>0 to enable experience replay. Saved samples will be replayed with a p:1 proportion to new data samples.
	##########################################
	"prioritized_replay": True,
	"learning_starts": 100, # How many batches to sample before learning starts. Every batch has size 'rollout_fragment_length' (default is 50).
	"buffer_options": {
		'priority_id': "gains", # Which batch column to use for prioritisation. One of the following: gains, importance_weights, unweighted_advantages, advantages, rewards, prev_rewards, action_logp.
		'priority_aggregation_fn': 'np.sum', # A reduce function that takes as input a list of numbers and returns a number representing a batch priority.
		'cluster_size': None, # Maximum number of batches stored in a cluster (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'rollout_fragment_length' (default is 50).
		'global_size': 2**13, # Maximum number of batches stored in all clusters (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'rollout_fragment_length' (default is 50).
		'alpha': 0.5, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'beta': None, # Parameter that regulates a mechanism for computing importance sampling; PPO probably does not need it because it has another mechanism for importance weighting.
		'epsilon': 1e-6, # Epsilon to add to a priority so that it is never equal to 0.
		'prioritized_drop_probability': 0, # Probability of dropping the batch having the lowest priority in the buffer.
		'update_insertion_time_when_sampling': False, # Whether to update the insertion time batches to the time of sampling. It requires prioritized_drop_probability < 1. In DQN default is False.
		'global_distribution_matching': False, # Whether to use a random number rather than the batch priority during prioritised dropping. If True then: At time t the probability of any experience being the max experience is 1/t regardless of when the sample was added, guaranteeing that (when prioritized_drop_probability==1) at any given time the sampled experiences will approximately match the distribution of all samples seen so far.
		'prioritised_cluster_sampling_strategy': 'highest', # Whether to select which cluster to replay in a prioritised fashion. Four options: None; 'highest' - clusters with the highest priority are more likely to be sampled; 'average' - prioritise the cluster with priority closest to the average cluster priority; 'above_average' - prioritise the cluster with priority closest to the cluster with the smallest priority greater than the average cluster priority.
	},
	"clustering_scheme": "moving_best_extrinsic_reward_with_multiple_types", # Which scheme to use for building clusters. One of the following: none, extrinsic_reward, moving_best_extrinsic_reward, moving_best_extrinsic_reward_with_type, reward_with_type, reward_with_multiple_types, moving_best_extrinsic_reward_with_multiple_types.
	"update_only_sampled_cluster": True, # Whether to update the priority only in the sampled cluster and not in all, if the same batch is in more than one cluster. Setting this option to True causes a slighlty higher memory consumption but shall increase by far the speed in updating priorities.
	"gae_with_vtrace": False, # Formula for computing the advantages: combines GAE with V-Trace, for better sample efficiency.
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
		rollout["importance_weights"][::-1]
	)
	rollout[Postprocessing.ADVANTAGES] = np.array(reversed_cumulative_advantage, dtype=np.float32)[::-1]
	rollout[Postprocessing.VALUE_TARGETS] = np.array(reversed_cumulative_return, dtype=np.float32)[::-1]
	assert all(val.shape[0] == rollout_size for key, val in rollout.items()), "Rollout stacked incorrectly!"
	return rollout

def xappo_postprocess_trajectory(policy, sample_batch, other_agent_batches=None, episode=None):
	# Add PPO's importance weights
	_,_,actions_info = policy.compute_actions_from_input_dict(sample_batch, episodes=[episode])
	action_logp = actions_info['action_logp']
	old_action_logp = sample_batch[SampleBatch.ACTION_LOGP]
	logp_ratio = np.exp(action_logp - old_action_logp)
	sample_batch["importance_weights"] = logp_ratio
	# Add advantages, after "importance_weights"
	completed = sample_batch["dones"][-1]
	if completed:
		last_r = 0.0
	else:
		next_state = []
		for i in range(policy.num_state_tensors()):
			next_state.append([sample_batch["state_out_{}".format(i)][-1]])
		last_r = policy._value(sample_batch[SampleBatch.NEXT_OBS][-1],
							   sample_batch[SampleBatch.ACTIONS][-1],
							   sample_batch[SampleBatch.REWARDS][-1],
							   *next_state)
	if policy.config["gae_with_vtrace"]:
		sample_batch = compute_gae_v_advantages(sample_batch, last_r, policy.config["gamma"], policy.config["lambda"])
	else:
		sample_batch = compute_advantages(
			sample_batch,
			last_r,
			policy.config["gamma"],
			policy.config["lambda"],
			use_gae=policy.config["use_gae"],
			use_critic=policy.config["use_critic"]
		)
	sample_batch['unweighted_'+Postprocessing.ADVANTAGES] = sample_batch[Postprocessing.ADVANTAGES].copy()
	if 'weights' in sample_batch:
		# print(sample_batch['weights'])
		sample_batch[Postprocessing.ADVANTAGES] *= sample_batch['weights']
	# Add gains
	advantages = sample_batch[Postprocessing.ADVANTAGES]
	new_priorities = advantages * logp_ratio
	sample_batch["gains"] = new_priorities
	# if "new_obs" in sample_batch:
	# 	del sample_batch.data["new_obs"]  # not used, so save some bandwidth
	return sample_batch

XAPPOTFPolicy = AsyncPPOTFPolicy.with_updates(
	name="XAPPOTFPolicy",
	extra_action_fetches_fn=vf_preds_fetches,
	postprocess_fn=xappo_postprocess_trajectory,
)
XAPPOTorchPolicy = AsyncPPOTorchPolicy.with_updates(
	name="XAPPOTorchPolicy",
	extra_action_out_fn=vf_preds_fetches,
	postprocess_fn=xappo_postprocess_trajectory,
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

	def update_replayed_fn(samples):
		if isinstance(samples, MultiAgentBatch):
			for pid, batch in samples.policy_batches.items():
				if pid not in local_worker.policies_to_train:
					continue
				policy = local_worker.policy_map[pid]
				samples.policy_batches[pid] = xappo_postprocess_trajectory(policy, batch)
			local_replay_buffer.update_priorities(samples.policy_batches)
		else:
			samples = xappo_postprocess_trajectory(local_worker.policy_map[DEFAULT_POLICY_ID], samples)
			local_replay_buffer.update_priorities({DEFAULT_POLICY_ID:samples})
		return samples

	# Augment with replay and concat to desired train batch size.
	train_batches = rollouts \
		.for_each(lambda batch: batch.decompress_if_needed()) \
		.for_each(lambda batch: assign_types(batch, clustering_scheme, config["rollout_fragment_length"])) \
		.flatten() \
		.for_each(MixInReplay(
			local_buffer=local_replay_buffer,
			replay_proportion=config["replay_proportion"],
			update_replayed_fn=update_replayed_fn,
		)) \
		.flatten() \
		.combine(ConcatBatches(min_batch_size=config["train_batch_size"]))

	# Start the learner thread.
	learner_thread = make_learner_thread(local_worker, config)
	learner_thread.start()

	# This sub-flow sends experiences to the learner.
	enqueue_op = train_batches.for_each(Enqueue(learner_thread.inqueue)) 
	# Only need to update workers if there are remote workers.
	if workers.remote_workers():
		enqueue_op = enqueue_op.zip_with_source_actor() \
			.for_each(BroadcastUpdateLearnerWeights(
				learner_thread, workers,
				broadcast_interval=config["broadcast_interval"]))

	dequeue_op = Dequeue(learner_thread.outqueue, check=learner_thread.is_alive) \
		.for_each(record_steps_trained)

	merged_op = Concurrently([enqueue_op, dequeue_op], mode="async", output_indexes=[1])

	# Callback for APPO to use to update KL, target network periodically.
	# The input to the callback is the learner fetches dict.
	if config["after_train_step"]:
		merged_op = merged_op \
			.for_each(lambda t: t[1]) \
			.for_each(config["after_train_step"](workers, config))

	return StandardMetricsReporting(merged_op, workers, config).for_each(learner_thread.add_learner_metrics)

XAPPOTrainer = APPOTrainer.with_updates(
	name="XAPPO", 
	default_config=XAPPO_DEFAULT_CONFIG,
	default_policy=XAPPOTFPolicy,
	get_policy_class=xappo_get_policy_class,
	execution_plan=xappo_execution_plan,
)
