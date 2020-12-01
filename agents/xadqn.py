"""
XADQN - eXplanation-Aware Deep Q-Networks (DQN, Rainbow, Parametric DQN)
==============================================

Detailed documentation:
https://docs.ray.io/en/master/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn
"""  # noqa: E501

from ray.rllib.agents.dqn.dqn import calculate_rr_weights, DQNTrainer, TrainOneStep, UpdateTargetNetwork, Concurrently, StandardMetricsReporting, LEARNER_STATS_KEY, DEFAULT_CONFIG as DQN_DEFAULT_CONFIG
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy, compute_q_values as torch_compute_q_values, torch
from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy, compute_q_values as tf_compute_q_values, tf
from ray.rllib.utils.tf_ops import explained_variance as tf_explained_variance
from ray.rllib.utils.torch_ops import explained_variance as torch_explained_variance
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID

from agents.xa_ops import *
from experience_buffers.replay_ops import StoreToReplayBuffer, Replay

XADQN_DEFAULT_CONFIG = DQN_DEFAULT_CONFIG
# For more config options, see here: https://docs.ray.io/en/master/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn
XADQN_DEFAULT_CONFIG["prioritized_replay"] = True
XADQN_DEFAULT_CONFIG["buffer_options"] = {
	'priority_id': "weights", # What batch column to use for prioritisation. One of the following: rewards, prev_rewards, weights
	'priority_aggregation_fn': 'lambda x: np.mean(np.abs(x))', # A reduce function that takes as input a list of numbers and returns a number representing a batch's priority
	'size': 50000, # "Maximum number of batches stored in the experience buffer."
	'alpha': 0.6, # "How much prioritization is used (0 - no prioritization, 1 - full prioritization)."
	'beta': 0.4, # Parameter that regulates a mechanism for computing importance sampling.
	'epsilon': 1e-6, # Epsilon to add to the TD errors when updating priorities.
	'prioritized_drop_probability': 1, # Probability of dropping experience with the lowest priority in the buffer
	'global_distribution_matching': False, # "If True, then: At time t the probability of any experience being the max experience is 1/t regardless of when the sample was added, guaranteeing that at any given time the sampled experiences will approximately match the distribution of all samples seen so far."
	'prioritised_cluster_sampling': True, # Whether to select which cluster to replay in a prioritised fashion
}
XADQN_DEFAULT_CONFIG["clustering_scheme"] = "moving_best_extrinsic_reward_with_type" # Which scheme to use for building clusters. One of the following: none, extrinsic_reward, moving_best_extrinsic_reward, moving_best_extrinsic_reward_with_type, reward_with_type
XADQN_DEFAULT_CONFIG["batch_mode"] = "complete_episodes" # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes' otherwise it can also be 'truncate_episodes'

########################
# XADQN's Policy
########################

def tf_get_selected_qts(policy, train_batch):
	config = policy.config
	# q network evaluation
	q_t, _, _ = tf_compute_q_values(policy, policy.q_model, train_batch[SampleBatch.CUR_OBS], explore=False)
	# target q network evalution
	q_tp1, _, _ = tf_compute_q_values(policy, policy.target_q_model, train_batch[SampleBatch.NEXT_OBS], explore=False)
	# q scores for actions which we know were selected in the given state.
	one_hot_selection = tf.one_hot(tf.cast(train_batch[SampleBatch.ACTIONS], tf.int32), policy.action_space.n)
	q_t_selected = tf.reduce_sum(q_t * one_hot_selection, 1)
	# compute estimate of best possible value starting from state at t + 1
	if config["double_q"]:
		q_tp1_using_online_net, _, _ = tf_compute_q_values(policy, policy.q_model, train_batch[SampleBatch.NEXT_OBS], explore=False)
		q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
		q_tp1_best_one_hot_selection = tf.one_hot(q_tp1_best_using_online_net, policy.action_space.n)
		q_tp1_best = tf.reduce_sum(q_tp1 * q_tp1_best_one_hot_selection, 1)
	else:
		q_tp1_best_one_hot_selection = tf.one_hot(tf.argmax(q_tp1, 1), policy.action_space.n)
		q_tp1_best = tf.reduce_sum(q_tp1 * q_tp1_best_one_hot_selection, 1)
	gamma = config["gamma"]
	n_step = config["n_step"]
	rewards = train_batch[SampleBatch.REWARDS]
	done_mask = tf.cast(train_batch[SampleBatch.DONES], tf.float32)
	q_tp1_best_masked = (1.0 - done_mask) * q_tp1_best
	q_t_selected_target = rewards + gamma**n_step * q_tp1_best_masked
	return q_t_selected, q_t_selected_target

def torch_get_selected_qts(policy, train_batch):
	config = policy.config
	# Q-network evaluation.
	q_t, _, _ = torch_compute_q_values(policy, policy.q_model, train_batch[SampleBatch.CUR_OBS], explore=False, is_training=True)
	# Target Q-network evaluation.
	q_tp1, _, _ = torch_compute_q_values(policy, policy.target_q_model, train_batch[SampleBatch.NEXT_OBS], explore=False, is_training=True)
	# Q scores for actions which we know were selected in the given state.
	one_hot_selection = F.one_hot(train_batch[SampleBatch.ACTIONS].long(), policy.action_space.n)
	q_t_selected = torch.sum(torch.where(q_t > FLOAT_MIN, q_t, torch.tensor(0.0, device=policy.device)) * one_hot_selection, 1)
	# compute estimate of best possible value starting from state at t + 1
	if config["double_q"]:
		q_tp1_using_online_net, _, _ = torch_compute_q_values(policy, policy.q_model, train_batch[SampleBatch.NEXT_OBS], explore=False, is_training=True)
		q_tp1_best_using_online_net = torch.argmax(q_tp1_using_online_net, 1)
		q_tp1_best_one_hot_selection = F.one_hot(q_tp1_best_using_online_net, policy.action_space.n)
		q_tp1_best = torch.sum(torch.where(q_tp1 > FLOAT_MIN, q_tp1, torch.tensor(0.0, device=policy.device)) * q_tp1_best_one_hot_selection, 1)
	else:
		q_tp1_best_one_hot_selection = F.one_hot(torch.argmax(q_tp1, 1), policy.action_space.n)
		q_tp1_best = torch.sum(torch.where(q_tp1 > FLOAT_MIN, q_tp1, torch.tensor(0.0, device=policy.device)) * q_tp1_best_one_hot_selection, 1)
	gamma = config["gamma"]
	n_step = config["n_step"]
	rewards = train_batch[SampleBatch.REWARDS]
	done_mask = tf.cast(train_batch[SampleBatch.DONES], tf.float32)
	q_tp1_best_masked = (1.0 - done_mask) * q_tp1_best
	q_t_selected_target = rewards + gamma**n_step * q_tp1_best_masked
	return q_t_selected, q_t_selected_target

def build_xadqn_stats(policy, batch):
	mean_fn = torch.mean if policy.config["framework"]=="torch" else tf.reduce_mean
	explained_variance_fn = torch_explained_variance if policy.config["framework"]=="torch" else tf_explained_variance
	qts_fn = torch_get_selected_qts if policy.config["framework"]=="torch" else tf_get_selected_qts
	q_t_selected, q_t_selected_target = qts_fn(policy, batch)
	return dict({
		"cur_lr": tf.cast(policy.cur_lr, tf.float64),
		"vf_explained_var": mean_fn(explained_variance_fn(q_t_selected_target, q_t_selected)),
	}, **policy.q_loss.stats)

XADQNTFPolicy = DQNTFPolicy.with_updates(
	stats_fn=build_xadqn_stats,
)
XADQNTorchPolicy = DQNTorchPolicy.with_updates(
	stats_fn=build_xadqn_stats,
)

########################
# XADQN's Execution Plan
########################

def get_policy_class(config):
	if config["framework"] == "torch":
		return XADQNTorchPolicy
	return XADQNTFPolicy

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
	get_policy_class=get_policy_class,
)
