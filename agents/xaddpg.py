"""
XADDPG - eXplanation-Aware Deep Deterministic Policy Gradient
==============================================

Detailed documentation:
https://docs.ray.io/en/master/rllib-algorithms.html#deep-deterministic-policy-gradients-ddpg-td3
"""  # noqa: E501

from agents.xadqn import xadqn_execution_plan
from ray.rllib.agents.ddpg.ddpg import DDPGTrainer, DEFAULT_CONFIG as DDPG_DEFAULT_CONFIG
from ray.rllib.agents.ddpg.ddpg_tf_policy import DDPGTFPolicy, tf, PRIO_WEIGHTS
from ray.rllib.agents.ddpg.ddpg_torch_policy import DDPGTorchPolicy, torch
from ray.rllib.utils.tf_ops import explained_variance as tf_explained_variance
from ray.rllib.utils.torch_ops import explained_variance as torch_explained_variance
from ray.rllib.policy.sample_batch import SampleBatch

XADDPG_DEFAULT_CONFIG = DDPG_DEFAULT_CONFIG
# For more config options, see here: https://docs.ray.io/en/master/rllib-algorithms.html#deep-q-networks-dqn-rainbow-parametric-dqn
XADDPG_DEFAULT_CONFIG["prioritized_replay"] = True
XADDPG_DEFAULT_CONFIG["buffer_options"] = {
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
XADDPG_DEFAULT_CONFIG["clustering_scheme"] = "moving_best_extrinsic_reward_with_type" # Which scheme to use for building clusters. One of the following: none, extrinsic_reward, moving_best_extrinsic_reward, moving_best_extrinsic_reward_with_type, reward_with_type
XADDPG_DEFAULT_CONFIG["batch_mode"] = "complete_episodes" # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes' otherwise it can also be 'truncate_episodes'

########################
# XADDPG's Policy
########################

def tf_get_selected_qts(policy, model, train_batch):
	twin_q = policy.config["twin_q"]
	gamma = policy.config["gamma"]
	n_step = policy.config["n_step"]

	input_dict = {
		"obs": train_batch[SampleBatch.CUR_OBS],
		"is_training": True,
	}
	input_dict_next = {
		"obs": train_batch[SampleBatch.NEXT_OBS],
		"is_training": True,
	}

	model_out_t, _ = model(input_dict, [], None)
	target_model_out_tp1, _ = policy.target_model(input_dict_next, [], None)

	# Policy network evaluation.
	policy_tp1 = policy.target_model.get_policy_output(target_model_out_tp1)

	# Action outputs.
	if policy.config["smooth_target_policy"]:
		target_noise_clip = policy.config["target_noise_clip"]
		clipped_normal_sample = tf.clip_by_value(
			tf.random.normal(
				tf.shape(policy_tp1), stddev=policy.config["target_noise"]),
			-target_noise_clip, target_noise_clip)
		policy_tp1_smoothed = tf.clip_by_value(
			policy_tp1 + clipped_normal_sample,
			policy.action_space.low * tf.ones_like(policy_tp1),
			policy.action_space.high * tf.ones_like(policy_tp1))
	else:
		# No smoothing, just use deterministic actions.
		policy_tp1_smoothed = policy_tp1

	# Q-net(s) evaluation.
	# prev_update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
	# Q-values for given actions & observations in given current
	q_t = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])

	# Target q-net(s) evaluation.
	q_tp1 = policy.target_model.get_q_values(target_model_out_tp1, policy_tp1_smoothed)

	if twin_q:
		twin_q_tp1 = policy.target_model.get_twin_q_values(target_model_out_tp1, policy_tp1_smoothed)

	q_t_selected = tf.squeeze(q_t, axis=len(q_t.shape) - 1)
	if twin_q:
		q_tp1 = tf.minimum(q_tp1, twin_q_tp1)

	q_tp1_best = tf.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
	q_tp1_best_masked = (1.0 - tf.cast(train_batch[SampleBatch.DONES], tf.float32)) * q_tp1_best

	# Compute RHS of bellman equation.
	q_t_selected_target = train_batch[SampleBatch.REWARDS] + gamma**n_step * q_tp1_best_masked
	return q_t_selected, q_t_selected_target

def torch_get_selected_qts(policy, model, train_batch):
	twin_q = policy.config["twin_q"]
	gamma = policy.config["gamma"]
	n_step = policy.config["n_step"]

	input_dict = {
		"obs": train_batch[SampleBatch.CUR_OBS],
		"is_training": True,
	}
	input_dict_next = {
		"obs": train_batch[SampleBatch.NEXT_OBS],
		"is_training": True,
	}

	model_out_t, _ = model(input_dict, [], None)
	target_model_out_tp1, _ = policy.target_model(input_dict_next, [], None)

	policy_tp1 = policy.target_model.get_policy_output(target_model_out_tp1)

	# Action outputs.
	if policy.config["smooth_target_policy"]:
		target_noise_clip = policy.config["target_noise_clip"]
		clipped_normal_sample = torch.clamp(
			torch.normal(
				mean=torch.zeros(policy_tp1.size()),
				std=policy.config["target_noise"]).to(policy_tp1.device),
			-target_noise_clip, target_noise_clip)

		policy_tp1_smoothed = torch.min(
			torch.max(
				policy_tp1 + clipped_normal_sample,
				torch.tensor(
					policy.action_space.low,
					dtype=torch.float32,
					device=policy_tp1.device)),
			torch.tensor(
				policy.action_space.high,
				dtype=torch.float32,
				device=policy_tp1.device))
	else:
		# No smoothing, just use deterministic actions.
		policy_tp1_smoothed = policy_tp1

	# Q-net(s) evaluation.
	# prev_update_ops = set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS))
	# Q-values for given actions & observations in given current
	q_t = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])

	# Target q-net(s) evaluation.
	q_tp1 = policy.target_model.get_q_values(target_model_out_tp1, policy_tp1_smoothed)

	if twin_q:
		twin_q_tp1 = policy.target_model.get_twin_q_values(target_model_out_tp1, policy_tp1_smoothed)

	q_t_selected = torch.squeeze(q_t, axis=len(q_t.shape) - 1)
	if twin_q:
		q_tp1 = torch.min(q_tp1, twin_q_tp1)

	q_tp1_best = torch.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
	q_tp1_best_masked = (1.0 - train_batch[SampleBatch.DONES].float()) * q_tp1_best

	# Compute RHS of bellman equation.
	q_t_selected_target = (train_batch[SampleBatch.REWARDS] + gamma**n_step * q_tp1_best_masked).detach()

	return q_t_selected, q_t_selected_target

def build_xaddpg_stats(policy, batch):
	explained_variance_fn = torch_explained_variance if policy.config["framework"]=="torch" else tf_explained_variance
	mean_fn = torch.mean if policy.config["framework"]=="torch" else tf.reduce_mean
	max_fn = torch.max if policy.config["framework"]=="torch" else tf.reduce_max
	min_fn = torch.min if policy.config["framework"]=="torch" else tf.reduce_min
	qts_fn = torch_get_selected_qts if policy.config["framework"]=="torch" else tf_get_selected_qts
	q_t_selected, q_t_selected_target = qts_fn(policy, policy.model, batch)
	stats = {
		"actor_loss": policy.actor_loss,
		"critic_loss": policy.critic_loss,
		"mean_q": mean_fn(policy.q_t),
		"max_q": max_fn(policy.q_t),
		"min_q": min_fn(policy.q_t),
		"mean_td_error": mean_fn(policy.td_error),
		# "td_error": policy.td_error,
		"vf_explained_var": mean_fn(explained_variance_fn(q_t_selected_target, q_t_selected)),
	}
	return stats

XADDPGTFPolicy = DDPGTFPolicy.with_updates(
	stats_fn=build_xaddpg_stats,
)
XADDPGTorchPolicy = DDPGTorchPolicy.with_updates(
	stats_fn=build_xaddpg_stats,
)

########################
# XADDPG's Execution Plan
########################

def get_policy_class(config):
	if config["framework"] == "torch":
		return XADDPGTorchPolicy
	return XADDPGTFPolicy

XADDPGTrainer = DDPGTrainer.with_updates(
	name="XADDPG", 
	execution_plan=xadqn_execution_plan,
	get_policy_class=get_policy_class,
)
