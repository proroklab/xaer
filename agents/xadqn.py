"""
XADQN - eXplanation-Aware Deep Q-Networks (DQN, Rainbow, Parametric DQN)
==============================================

Detailed documentation:
https://docs.ray.io/en/master/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn
"""  # noqa: E501

from ray.rllib.agents.dqn.dqn import calculate_rr_weights, DQNTrainer, TrainOneStep, UpdateTargetNetwork, Concurrently, StandardMetricsReporting, LEARNER_STATS_KEY, DEFAULT_CONFIG as DQN_DEFAULT_CONFIG
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

from agents.xa_ops import *
from experience_buffers.replay_ops import StoreToReplayBuffer, Replay

XADQN_DEFAULT_CONFIG = DQN_DEFAULT_CONFIG
XADQN_DEFAULT_CONFIG["prioritized_replay"] = True
XADQN_DEFAULT_CONFIG["buffer_options"] = {
	'priority_id': "weights", # one of the following: gains, importance_weights, rewards, prev_rewards, action_logp
	'priority_aggregation_fn': 'lambda x: np.mean(np.abs(x))', # a reduce function (from a list of numbers to a number)
	'size': 50000, 
	'alpha': 0.6, 
	'beta': 0.4, 
	'epsilon': 1e-6, # Epsilon to add to the TD errors when updating priorities.
	'prioritized_drop_probability': 0, 
	'global_distribution_matching': False, 
	'prioritised_cluster_sampling': True, 
}
XADQN_DEFAULT_CONFIG["clustering_scheme"] = "moving_best_extrinsic_reward_with_type" # one of the following: none, extrinsic_reward, moving_best_extrinsic_reward, moving_best_extrinsic_reward_with_type, reward_with_type
XADQN_DEFAULT_CONFIG["batch_mode"] = "complete_episodes" # can be equal to 'truncate_episodes' only when 'clustering_scheme' is 'none'

def xadqn_execution_plan(workers, config):
	local_replay_buffer, clustering_scheme = get_clustered_replay_buffer(config)
	def update_priorities(item):
		batch, info_dict = item
		# print(info_dict, config.get("prioritized_replay"))
		if config.get("prioritized_replay"):
			priority_id = config["buffer_options"]["priority_id"]
			if priority_id == "weights":
				batch[priority_id] = info_dict[DEFAULT_POLICY_ID][LEARNER_STATS_KEY].get("td_error", info_dict[DEFAULT_POLICY_ID].get("td_error"))
			local_replay_buffer.update_priority(batch)
		return info_dict

	rollouts = ParallelRollouts(workers, mode="bulk_sync")

	# We execute the following steps concurrently:
	# (1) Generate rollouts and store them in our local replay buffer. Calling
	# next() on store_op drives this.
	store_op = rollouts \
		.for_each(lambda batch: batch.split_by_episode()) \
		.flatten() \
		.for_each(lambda episode: episode.timeslices(config["replay_sequence_length"])) \
		.for_each(lambda episode: assign_types_from_episode(episode, clustering_scheme)) \
		.flatten() \
		.for_each(StoreToReplayBuffer(local_buffer=local_replay_buffer))

	# (2) Read and train on experiences from the replay buffer. Every batch
	# returned from the LocalReplay() iterator is passed to TrainOneStep to
	# take a SGD step, and then we decide whether to update the target network.
	post_fn = config.get("before_learn_on_batch") or (lambda b, *a: b)
	replay_op = Replay(local_buffer=local_replay_buffer, replay_batch_size=config["train_batch_size"]) \
		.flatten() \
		.combine(ConcatBatches(min_batch_size=config["train_batch_size"])) \
		.for_each(lambda x: post_fn(x, workers, config)) \
		.for_each(TrainOneStep(workers)) \
		.for_each(update_priorities) \
		.for_each(UpdateTargetNetwork(workers, config["target_network_update_freq"]))

	# Alternate deterministically between (1) and (2). Only return the output
	# of (2) since training metrics are not available until (2) runs.
	train_op = Concurrently([store_op, replay_op], mode="round_robin", output_indexes=[1], round_robin_weights=calculate_rr_weights(config))

	return StandardMetricsReporting(train_op, workers, config)

XADQNTrainer = DQNTrainer.with_updates(
	name="XADQN", 
	execution_plan=xadqn_execution_plan,
)
