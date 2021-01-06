"""
XADQN - eXplanation-Aware Deep Q-Networks (DQN, Rainbow, Parametric DQN)
==============================================

Detailed documentation:
https://docs.ray.io/en/master/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn
"""  # noqa: E501
import numpy as np
from more_itertools import unique_everseen
from ray.rllib.agents.dqn.dqn import calculate_rr_weights, DQNTrainer, TrainOneStep, UpdateTargetNetwork, Concurrently, StandardMetricsReporting, LEARNER_STATS_KEY, DEFAULT_CONFIG as DQN_DEFAULT_CONFIG
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy, compute_q_values as torch_compute_q_values, torch, F, FLOAT_MIN
from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy, compute_q_values as tf_compute_q_values, tf, _adjust_nstep
from ray.rllib.utils.tf_ops import explained_variance as tf_explained_variance
from ray.rllib.utils.torch_ops import explained_variance as torch_explained_variance
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch, DEFAULT_POLICY_ID

from xarl.experience_buffers.replay_ops import StoreToReplayBuffer, Replay, get_clustered_replay_buffer, assign_types
from xarl.experience_buffers.replay_buffer import get_batch_infos, get_batch_uid

XADQN_EXTRA_OPTIONS = {
	"batch_mode": "complete_episodes", # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes', otherwise it can also be 'truncate_episodes'.
	"prioritized_replay": True,
	##########################################
	"buffer_options": {
		'priority_id': "td_errors", # Which batch column to use for prioritisation. Default is inherited by DQN and it is 'td_errors'. One of the following: rewards, prev_rewards, td_errors.
		'priority_aggregation_fn': 'lambda x: np.mean(np.abs(x))', # A reduction that takes as input a list of numbers and returns a number representing a batch priority.
		'cluster_size': None, # Default None, implying being equal to global_size. Maximum number of batches stored in a cluster (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'replay_sequence_length' (default is 1).
		'global_size': 2**15, # Default 50000. Maximum number of batches stored in all clusters (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'replay_sequence_length' (default is 1).
		'alpha': 0.6, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'beta': 0.4, # To what degree to use importance weights (0 - no corrections, 1 - full correction).
		'eta': None, # A value > 0 that enables eta-weighting, thus allowing for importance weighting with priorities lower than 0 if beta is > 0. Eta is used to avoid importance weights equal to 0 when the sampled batch is the one with the highest priority. The closer eta is to 0, the closer to 0 would be the importance weight of the highest-priority batch.
		'epsilon': 1e-6, # Epsilon to add to a priority so that it is never equal to 0.
		'prioritized_drop_probability': 0, # Probability of dropping the batch having the lowest priority in the buffer instead of the one having the lowest timestamp. In DQN default is 0.
		'update_insertion_time_when_sampling': False, # Whether to update the insertion time batches to the time of sampling. It requires prioritized_drop_probability < 1. In DQN default is False.
		'global_distribution_matching': False, # Whether to use a random number rather than the batch priority during prioritised dropping. If True then: At time t the probability of any experience being the max experience is 1/t regardless of when the sample was added, guaranteeing that (when prioritized_drop_probability==1) at any given time the sampled experiences will approximately match the distribution of all samples seen so far.
		'prioritised_cluster_sampling_strategy': 'highest', # Whether to select which cluster to replay in a prioritised fashion -- 4 options: None; 'highest' - clusters with the highest priority are more likely to be sampled; 'average' - prioritise the cluster with priority closest to the average cluster priority; 'above_average' - prioritise the cluster with priority closest to the cluster with the smallest priority greater than the average cluster priority.
		'cluster_level_weighting': False, # Whether to use only cluster-level information to compute importance weights rather than the whole buffer.
	},
	"clustering_scheme": "multiple_types", # Which scheme to use for building clusters. One of the following: "none", "reward_against_zero", "reward_against_mean", "multiple_types_with_reward_against_mean", "multiple_types_with_reward_against_zero", "type_with_reward_against_mean", "multiple_types", "type".
	"cluster_with_episode_type": False, # Most useful with sparse-reward environments. Whether to cluster experience using information at episode-level.
	"cluster_overview_size": 1, # cluster_overview_size <= train_batch_size. If None, then cluster_overview_size is automatically set to train_batch_size. -- When building a single train batch, do not sample a new cluster before x batches are sampled from it. The closer cluster_overview_size is to train_batch_size, the faster is the batch sampling procedure.
	"update_only_sampled_cluster": True, # If True, when sampling a batch from a cluster, no changes/updates to other clusters are performed if that batch is shared among these other clusters. Enabling this option would slightly speed-up batch sampling, by a constant proportional to the number of different clusters in the buffer.
}
# The combination of update_insertion_time_when_sampling==True and prioritized_drop_probability==0 helps mantaining in the buffer only those batches with the most up-to-date priorities.
XADQN_DEFAULT_CONFIG = DQNTrainer.merge_trainer_configs(
	DQN_DEFAULT_CONFIG, # For more details, see here: https://docs.ray.io/en/master/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn
	XADQN_EXTRA_OPTIONS,
	_allow_unknown_configs=True
)

########################
# XADQN's Policy
########################

def xa_postprocess_nstep_and_prio(policy, batch, other_agent=None, episode=None):
	# N-step Q adjustments.
	if policy.config["n_step"] > 1:
		_adjust_nstep(policy.config["n_step"], policy.config["gamma"], batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS], batch[SampleBatch.REWARDS], batch[SampleBatch.NEXT_OBS], batch[SampleBatch.DONES])
	if 'weights' not in batch:
		batch['weights'] = np.ones_like(batch[SampleBatch.REWARDS])
	batch.data["td_errors"] = policy.compute_td_error(batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS], batch[SampleBatch.REWARDS], batch[SampleBatch.NEXT_OBS], batch[SampleBatch.DONES], batch['weights'])
	return batch

XADQNTFPolicy = DQNTFPolicy.with_updates(
	name="XADQNTFPolicy",
	postprocess_fn=xa_postprocess_nstep_and_prio,
)
XADQNTorchPolicy = DQNTorchPolicy.with_updates(
	name="XADQNTorchPolicy",
	postprocess_fn=xa_postprocess_nstep_and_prio,
)

def get_policy_class(config):
	if config["framework"] == "torch": return XADQNTorchPolicy
	return XADQNTFPolicy

########################
# XADQN's Execution Plan
########################

def xadqn_execution_plan(workers, config):
	replay_batch_size = config["train_batch_size"]
	replay_sequence_length = config["replay_sequence_length"]
	if replay_sequence_length and replay_sequence_length > 1:
		replay_batch_size = int(max(1, replay_batch_size // replay_sequence_length))
	local_replay_buffer, clustering_scheme = get_clustered_replay_buffer(config)
	local_worker = workers.local_worker()

	rollouts = ParallelRollouts(workers, mode="async")

	# We execute the following steps concurrently:
	# (1) Generate rollouts and store them in our local replay buffer. Calling
	# next() on store_op drives this.
	def store_batch(batch):
		sub_batch_list = assign_types(batch, clustering_scheme, replay_sequence_length, with_episode_type=config["cluster_with_episode_type"])
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
		if priority_id == "td_errors":
			for policy_id, info in info_dict.items():
				samples.policy_batches[policy_id]["td_errors"] = info.get("td_error", info[LEARNER_STATS_KEY].get("td_error"))
		# IMPORTANT: split train-batch into replay-batches, using batch_uid, before updating priorities
		policy_batch_list = []
		for policy_id, batch in samples.policy_batches.items():
			sub_batch_indexes = [
				i
				for i,infos in enumerate(batch['infos'])
				if "batch_uid" in infos
			] + [batch.count]
			sub_batch_iter = (
				batch.slice(sub_batch_indexes[j], sub_batch_indexes[j+1])
				for j in range(len(sub_batch_indexes)-1)
			)
			sub_batch_iter = unique_everseen(sub_batch_iter, key=get_batch_uid)
			for i,sub_batch in enumerate(sub_batch_iter):
				if i >= len(policy_batch_list):
					policy_batch_list.append({})
				policy_batch_list[i][policy_id] = sub_batch
		for policy_batch in policy_batch_list:
			local_replay_buffer.update_priorities(policy_batch)
		return info_dict
	post_fn = config.get("before_learn_on_batch") or (lambda b, *a: b)
	replay_op = Replay(
			local_buffer=local_replay_buffer, 
			replay_batch_size=replay_batch_size, 
			cluster_overview_size=config["cluster_overview_size"]
		) \
		.flatten() \
		.combine(ConcatBatches(min_batch_size=replay_batch_size)) \
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
	get_policy_class=get_policy_class,
)
