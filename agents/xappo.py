import collections

from ray.rllib.agents.impala.impala import *
from ray.rllib.agents.ppo.appo import *
from ray.rllib.agents.ppo.appo_tf_policy import *
from ray.rllib.agents.ppo.ppo_tf_policy import vf_preds_fetches
from ray.rllib.agents.ppo.appo_torch_policy import AsyncPPOTorchPolicy
# from ray.rllib.evaluation.postprocessing import discount_cumsum

from experience_buffers.replay_buffer import LocalReplayBuffer
# from experience_buffers.replay_ops import Replay, StoreToReplayBuffer
from experience_buffers.replay_ops import MixInReplay
from experience_buffers.clustering_scheme import *
from utils.misc import accumulate

IMPORTANCE_WEIGHTS = "importance_weight"
PRIO_WEIGHTS = "gain"
XAPPO_DEFAULT_CONFIG = DEFAULT_CONFIG
XAPPO_DEFAULT_CONFIG["worker_side_prioritization"] = True
XAPPO_DEFAULT_CONFIG["prioritized_replay"] = True
XAPPO_DEFAULT_CONFIG["buffer_options"] = {
	'size': 2**9, 
	'alpha': 0.5, 
	'prioritized_drop_probability': 0.5, 
	'global_distribution_matching': False, 
	'prioritised_cluster_sampling': True,
}
XAPPO_DEFAULT_CONFIG["replay_sequence_length"] = 1
XAPPO_DEFAULT_CONFIG["priority_weights_aggregator"] = 'np.mean'
XAPPO_DEFAULT_CONFIG["replay_proportion"] = 1
# How many steps of the model to sample before learning starts.
XAPPO_DEFAULT_CONFIG["learning_starts"] = 1000
XAPPO_DEFAULT_CONFIG["batch_mode"] = "complete_episodes"
XAPPO_DEFAULT_CONFIG["vtrace"] = False # batch_mode==complete_episodes implies vtrace==False
XAPPO_DEFAULT_CONFIG["clustering_scheme"] = "moving_best_extrinsic_reward_with_type"
XAPPO_DEFAULT_CONFIG["gae_with_vtrace"] = True

########################
# XAPPO's Trajectory Post-Processing
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
    assert SampleBatch.VF_PREDS in rollout, "use_critic=True but values not found"
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
	if policy.config["worker_side_prioritization"]:
		sample_batch.data[PRIO_WEIGHTS] = new_priorities
	else:
		sample_batch[PRIO_WEIGHTS] = new_priorities
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
	assert config["batch_mode"] == "complete_episodes", "XAPPO requires 'complete_episodes' as batch_mode"
	local_replay_buffer = LocalReplayBuffer(
		prioritized_replay=config["prioritized_replay"],
		priority_weights_key=PRIO_WEIGHTS,
		buffer_options=config["buffer_options"], 
		learning_starts=config["learning_starts"], 
		replay_sequence_length=config["replay_sequence_length"], 
		priority_weights_aggregator=config["priority_weights_aggregator"], 
	)
	clustering_scheme = eval(config["clustering_scheme"])()
	def update_prio(batch):
		if not batch:
			return
		if config.get("prioritized_replay"):
			samples = batch.data if config["worker_side_prioritization"] else batch
			local_replay_buffer.update_priority(
				batch_index=samples["batch_indexes"][0], 
				weights=samples[PRIO_WEIGHTS], 
				type_id=samples["batch_types"][0],
			)
		return batch
	def assign_types_from_episode(episode):
		episode_type = clustering_scheme.get_episode_type(episode)
		for batch in episode:
			batch_type = clustering_scheme.get_batch_type(batch, episode_type)
			batch["batch_types"] = np.array([batch_type]*batch.count)
		return episode

	rollouts = ParallelRollouts(workers, mode="async", num_async=config["max_sample_requests_in_flight_per_worker"])

	# Augment with replay and concat to desired train batch size.
	train_batches = rollouts \
		.for_each(lambda batch: batch.decompress_if_needed()) \
		.for_each(lambda batch: batch.split_by_episode()) \
		.flatten() \
		.for_each(lambda batch: batch.timeslices(config["rollout_fragment_length"])) \
		.for_each(assign_types_from_episode) \
		.flatten() \
		.for_each(MixInReplay(
			local_buffer=local_replay_buffer,
			replay_proportion=config["replay_proportion"])) \
		.flatten() \
		.combine(ConcatBatches(min_batch_size=config["train_batch_size"]))

	# Start the learner thread.
	learner_thread = make_learner_thread(workers.local_worker(), config)
	learner_thread.start()

	# This sub-flow sends experiences to the learner.
	enqueue_op = train_batches \
		.for_each(Enqueue(learner_thread.inqueue)) \
		.for_each(update_prio)
	# Only need to update workers if there are remote workers.
	if workers.remote_workers():
		enqueue_op = enqueue_op.zip_with_source_actor() \
			.for_each(BroadcastUpdateLearnerWeights(
				learner_thread, workers,
				broadcast_interval=config["broadcast_interval"]))

	# This sub-flow updates the steps trained counter based on learner output.
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
	default_policy=XAPPOTFPolicy,
	get_policy_class=xappo_get_policy_class,
	execution_plan=xappo_execution_plan,
)
