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

from xarl.agents.xa_ops import *
from xarl.experience_buffers.replay_ops import MixInReplay
from xarl.utils.misc import accumulate

IMPORTANCE_WEIGHTS = "importance_weights"
GAINS = "gains"

XAPPO_DEFAULT_CONFIG = APPOTrainer.merge_trainer_configs(
	DEFAULT_CONFIG, # For more details, see here: https://docs.ray.io/en/master/rllib-algorithms.html#asynchronous-proximal-policy-optimization-appo
	{
		"replay_proportion": 1, # Set a p>0 to enable experience replay. Saved samples will be replayed with a p:1 proportion to new data samples.
		"learning_starts": 1000, # How many batches to sample before learning starts.
		"prioritized_replay": True,
		"buffer_options": {
			'priority_id': GAINS, # Which batch column to use for prioritisation. One of the following: gains, importance_weights, advantages, rewards, prev_rewards, action_logp
			'priority_aggregation_fn': 'np.sum', # A reduce function that takes as input a list of numbers and returns a number representing a batch's priority
			'size': 2**9, # "Maximum number of batches stored in the experience buffer."
			'alpha': 0.5, # "How much prioritization is used (0 - no prioritization, 1 - full prioritization)."
			'beta': None, # Parameter that regulates a mechanism for computing importance sampling. Not needed in PPO.
			'epsilon': 1e-6, # Epsilon to add to the TD errors when updating priorities.
			'prioritized_drop_probability': 0.5, # Probability of dropping experience with the lowest priority in the buffer
			'global_distribution_matching': False, # "If True, then: At time t the probability of any experience being the max experience is 1/t regardless of when the sample was added, guaranteeing that at any given time the sampled experiences will approximately match the distribution of all samples seen so far."
			'prioritised_cluster_sampling': True, # Whether to select which cluster to replay in a prioritised fashion
		},
		"clustering_scheme": "moving_best_extrinsic_reward_with_multiple_types", # Which scheme to use for building clusters. One of the following: none, extrinsic_reward, moving_best_extrinsic_reward, moving_best_extrinsic_reward_with_type, reward_with_type, reward_with_multiple_types, moving_best_extrinsic_reward_with_multiple_types
		"batch_mode": "complete_episodes", # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes' otherwise it can also be 'truncate_episodes'
		"vtrace": False, # Formula for computing the advantage: batch_mode==complete_episodes implies vtrace==False
		"gae_with_vtrace": True, # Formula for computing the advantage: combines GAE with V-Tracing
	},
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
		rollout[IMPORTANCE_WEIGHTS][::-1]
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
	sample_batch[IMPORTANCE_WEIGHTS] = logp_ratio
	# Add advantages, after IMPORTANCE_WEIGHTS
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
	# Add gains
	advantages = sample_batch[Postprocessing.ADVANTAGES]
	new_priorities = advantages * logp_ratio
	sample_batch[GAINS] = new_priorities
	del sample_batch.data["new_obs"]  # not used, so save some bandwidth
	return sample_batch

XAPPOTFPolicy = AsyncPPOTFPolicy.with_updates(
	extra_action_fetches_fn=vf_preds_fetches,
	postprocess_fn=xappo_postprocess_trajectory,
)
XAPPOTorchPolicy = AsyncPPOTorchPolicy.with_updates(
	extra_action_out_fn=vf_preds_fetches,
	postprocess_fn=xappo_postprocess_trajectory,
)

########################
# XAPPO's Execution Plan
########################

def xappo_get_policy_class(config):
	if config["framework"] == "torch":
		return XAPPOTorchPolicy
	# return XAPPOTFPolicy

def xappo_execution_plan(workers, config):
	local_replay_buffer, clustering_scheme = get_clustered_replay_buffer(
		config,
		replay_batch_size=1,
		replay_sequence_length=None,
	)
	rollouts = ParallelRollouts(workers, mode="async", num_async=config["max_sample_requests_in_flight_per_worker"])

	# Augment with replay and concat to desired train batch size.
	train_batches = rollouts \
		.for_each(lambda batch: batch.decompress_if_needed()) \
		.for_each(lambda batch: batch.split_by_episode()) \
		.flatten() \
		.for_each(lambda episode: episode.timeslices(config["rollout_fragment_length"])) \
		.for_each(lambda episode: assign_types_to_episode(episode, clustering_scheme)) \
		.flatten() \
		.for_each(MixInReplay(
			local_buffer=local_replay_buffer,
			replay_proportion=config["replay_proportion"])) \
		.flatten() \
		.combine(ConcatBatches(min_batch_size=config["train_batch_size"]))

	# Start the learner thread.
	learner_thread = xa_make_learner_thread(workers.local_worker(), config)
	learner_thread.start()

	# This sub-flow sends experiences to the learner.
	enqueue_op = train_batches.for_each(Enqueue(learner_thread.inqueue)) 
	# Only need to update workers if there are remote workers.
	if workers.remote_workers():
		enqueue_op = enqueue_op.zip_with_source_actor() \
			.for_each(BroadcastUpdateLearnerWeights(
				learner_thread, workers,
				broadcast_interval=config["broadcast_interval"]))

	# This sub-flow updates the steps trained counter based on learner output.
	def update_priorities(item):
		if config.get("prioritized_replay"):
			samples, info_dict = item
			prio_dict = {}
			for policy_id, info in info_dict.items():
				batch = samples.policy_batches[policy_id]
				prio_dict[policy_id] = batch
			local_replay_buffer.update_priorities(prio_dict)
		return item
	dequeue_op = Dequeue(learner_thread.outqueue, check=learner_thread.is_alive) \
		.for_each(update_priorities) \
		.for_each(lambda x: (x[0].count, x[1])) \
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
