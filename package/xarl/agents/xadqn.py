"""
XADQN - eXplanation-Aware Deep Q-Networks (DQN, Rainbow, Parametric DQN)
==============================================

Detailed documentation:
https://docs.ray.io/en/master/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn
"""  # noqa: E501
from more_itertools import unique_everseen
from ray.rllib.agents.dqn.dqn import calculate_rr_weights, DQNTrainer, TrainOneStep, UpdateTargetNetwork, Concurrently, StandardMetricsReporting, LEARNER_STATS_KEY, DEFAULT_CONFIG as DQN_DEFAULT_CONFIG
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy, compute_q_values as torch_compute_q_values, torch, F, FLOAT_MIN
from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy, compute_q_values as tf_compute_q_values, tf
from ray.rllib.utils.tf_ops import explained_variance as tf_explained_variance
from ray.rllib.utils.torch_ops import explained_variance as torch_explained_variance
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches
from ray.rllib.policy.sample_batch import SampleBatch

from xarl.experience_buffers.replay_ops import StoreToReplayBuffer, Replay, get_clustered_replay_buffer, assign_types

XADQN_EXTRA_OPTIONS = {
	"rollout_fragment_length": 1,
	"train_batch_size": 2**8,
	"learning_starts": 1500,
	"grad_clip": None,
	"prioritized_replay": True,
	####################################
	"buffer_options": {
		'priority_id': "weights", # Which batch column to use for prioritisation. Default is inherited by DQN and it is 'weights'. One of the following: rewards, prev_rewards, weights.
		'priority_aggregation_fn': 'lambda x: np.mean(np.abs(x))', # A reduce function that takes as input a list of numbers and returns a number representing a batch priority.
		'cluster_size': 2**16, # Default 50000. Maximum number of batches stored in a cluster (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'replay_sequence_length' (default is 1).
		'global_size': 2**16, # Default 50000. Maximum number of batches stored in all clusters (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'replay_sequence_length' (default is 1).
		'alpha': 0.6, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'beta': 0.4, # Parameter that regulates a mechanism for computing importance sampling.
		'epsilon': 1e-6, # Epsilon to add to a priority so that it is never equal to 0.
		'prioritized_drop_probability': 0.5, # Probability of dropping the batch having the lowest priority in the buffer. In DQN default is 0.
		'global_distribution_matching': False, # Whether to use a random number rather than the batch priority during prioritised dropping. If True then: At time t the probability of any experience being the max experience is 1/t regardless of when the sample was added, guaranteeing that (when prioritized_drop_probability==1) at any given time the sampled experiences will approximately match the distribution of all samples seen so far.
		'prioritised_cluster_sampling': True, # Whether to select which cluster to replay in a prioritised fashion.
		'sample_simplest_unknown_task': 'average', # Whether to sample the simplest unknown task with higher probability. Three options: None; 'average' - the one with the cluster priority closest to the average cluster priority; 'above_average' - the one with the cluster priority closest to the cluster with the smallest priority greater than the average cluster priority. It requires prioritised_cluster_sampling==True.
	},
	"clustering_scheme": "moving_best_extrinsic_reward_with_multiple_types", # Which scheme to use for building clusters. One of the following: none, extrinsic_reward, moving_best_extrinsic_reward, moving_best_extrinsic_reward_with_type, reward_with_type, reward_with_multiple_types, moving_best_extrinsic_reward_with_multiple_types.
	"update_only_sampled_cluster": True, # Whether to update the priority only in the sampled cluster and not in all, if the same batch is in more than one cluster. Setting this option to True causes a slighlty higher memory consumption.
	"batch_mode": "complete_episodes", # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes', otherwise it can also be 'truncate_episodes'.
}
XADQN_DEFAULT_CONFIG = DQNTrainer.merge_trainer_configs(
	DQN_DEFAULT_CONFIG, # For more details, see here: https://docs.ray.io/en/master/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn
	XADQN_EXTRA_OPTIONS,
	_allow_unknown_configs=True
)

########################
# XADQN's Execution Plan
########################

def xadqn_execution_plan(workers, config):
	replay_batch_size = config["train_batch_size"]
	replay_sequence_length = config["replay_sequence_length"]
	if replay_sequence_length and replay_sequence_length > 1:
		replay_batch_size = int(max(1, replay_batch_size // replay_sequence_length))
	local_replay_buffer, clustering_scheme = get_clustered_replay_buffer(config)

	rollouts = ParallelRollouts(workers, mode="bulk_sync")

	# We execute the following steps concurrently:
	# (1) Generate rollouts and store them in our local replay buffer. Calling
	# next() on store_op drives this.
	def store_batch(batch):
		sub_batch_list = assign_types(batch, clustering_scheme, replay_sequence_length)
		store = StoreToReplayBuffer(local_buffer=local_replay_buffer)
		for sub_batch in sub_batch_list:
			store(sub_batch)
		return batch
	store_op = rollouts.for_each(store_batch)

	# (2) Read and train on experiences from the replay buffer. Every batch
	# returned from the LocalReplay() iterator is passed to TrainOneStep to
	# take a SGD step, and then we decide whether to update the target network.
	def update_priorities(item):
		samples, info_dict = item
		if not config.get("prioritized_replay"):
			return info_dict
		priority_id = config["buffer_options"]["priority_id"]
		if priority_id == "weights":
			for policy_id, info in info_dict.items():
				td_error = info.get("td_error", info[LEARNER_STATS_KEY].get("td_error"))
				samples.policy_batches[policy_id][priority_id] = td_error
		# IMPORTANT: split train-batch into replay-batches before updating priorities
		policy_batch_list = []
		for policy_id, batch in samples.policy_batches.items():
			sub_batch_iter = (
				sub_batch 
				for episode_batch in batch.split_by_episode()
				for sub_batch in episode_batch.timeslices(replay_sequence_length)
			) if replay_sequence_length > 1 else batch.timeslices(replay_sequence_length)
			sub_batch_iter = unique_everseen(sub_batch_iter, key=lambda x: tuple(x['infos'][0]["batch_index"].items()))
			for i,sub_batch in enumerate(sub_batch_iter):
				if i >= len(policy_batch_list):
					policy_batch_list.append({})
				policy_batch_list[i][policy_id] = sub_batch
		for policy_batch in policy_batch_list:
			local_replay_buffer.update_priorities(policy_batch)
		return info_dict
	post_fn = config.get("before_learn_on_batch") or (lambda b, *a: b)
	replay_op = Replay(local_buffer=local_replay_buffer, replay_batch_size=replay_batch_size) \
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
	default_config=XADQN_DEFAULT_CONFIG,
	execution_plan=xadqn_execution_plan,
)
