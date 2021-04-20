"""
TensorFlow policy class used for SAC.
"""

import gym
from gym.spaces import Box, Discrete
from functools import partial
import logging
from typing import Dict, List, Optional, Tuple, Type, Union

import ray
import ray.experimental.tf_utils
from ray.rllib.agents.dqn.dqn_tf_policy import postprocess_nstep_and_prio, PRIO_WEIGHTS
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import get_variable, try_import_tf, try_import_tfp
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.tf_ops import huber_loss
from ray.rllib.utils.typing import AgentID, LocalOptimizer, ModelGradients, TensorType, TrainerConfigDict

from ray.rllib.agents.sac.sac_tf_policy import _get_dist_class, TFActionDistribution, tf

def xasac_actor_critic_loss(policy: Policy, model: ModelV2, dist_class: Type[TFActionDistribution], train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
	# Should be True only for debugging purposes (e.g. test cases)!
	deterministic = policy.config["_deterministic_loss"]

	# Get the base model output from the train batch.
	model_out_t, _ = model({
		"obs": train_batch[SampleBatch.CUR_OBS],
		"is_training": policy._get_is_training_placeholder(),
	}, [], None)

	# Get the base model output from the next observations in the train batch.
	model_out_tp1, _ = model({
		"obs": train_batch[SampleBatch.NEXT_OBS],
		"is_training": policy._get_is_training_placeholder(),
	}, [], None)

	# Get the target model's base outputs from the next observations in the
	# train batch.
	target_model_out_tp1, _ = policy.target_model({
		"obs": train_batch[SampleBatch.NEXT_OBS],
		"is_training": policy._get_is_training_placeholder(),
	}, [], None)

	# Discrete actions case.
	if model.discrete:
		# Get all action probs directly from pi and form their logp.
		log_pis_t = tf.nn.log_softmax(model.get_policy_output(model_out_t), -1)
		policy_t = tf.math.exp(log_pis_t)
		log_pis_tp1 = tf.nn.log_softmax(
			model.get_policy_output(model_out_tp1), -1)
		policy_tp1 = tf.math.exp(log_pis_tp1)
		# Q-values.
		q_t = model.get_q_values(model_out_t)
		# Target Q-values.
		q_tp1 = policy.target_model.get_q_values(target_model_out_tp1)
		if policy.config["twin_q"]:
			twin_q_t = model.get_twin_q_values(model_out_t)
			twin_q_tp1 = policy.target_model.get_twin_q_values(
				target_model_out_tp1)
			q_tp1 = tf.reduce_min((q_tp1, twin_q_tp1), axis=0)
		q_tp1 -= model.alpha * log_pis_tp1

		# Actually selected Q-values (from the actions batch).
		one_hot = tf.one_hot(
			train_batch[SampleBatch.ACTIONS], depth=q_t.shape.as_list()[-1])
		q_t_selected = tf.reduce_sum(q_t * one_hot, axis=-1)
		if policy.config["twin_q"]:
			twin_q_t_selected = tf.reduce_sum(twin_q_t * one_hot, axis=-1)
		# Discrete case: "Best" means weighted by the policy (prob) outputs.
		q_tp1_best = tf.reduce_sum(tf.multiply(policy_tp1, q_tp1), axis=-1)
		q_tp1_best_masked = \
			(1.0 - tf.cast(train_batch[SampleBatch.DONES], tf.float32)) * \
			q_tp1_best
	# Continuous actions case.
	else:
		# Sample simgle actions from distribution.
		action_dist_class = _get_dist_class(policy.config, policy.action_space)
		action_dist_t = action_dist_class(
			model.get_policy_output(model_out_t), policy.model)
		policy_t = action_dist_t.sample() if not deterministic else \
			action_dist_t.deterministic_sample()
		log_pis_t = tf.expand_dims(action_dist_t.logp(policy_t), -1)
		action_dist_tp1 = action_dist_class(
			model.get_policy_output(model_out_tp1), policy.model)
		policy_tp1 = action_dist_tp1.sample() if not deterministic else \
			action_dist_tp1.deterministic_sample()
		log_pis_tp1 = tf.expand_dims(action_dist_tp1.logp(policy_tp1), -1)

		# Q-values for the actually selected actions.
		q_t = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
		if policy.config["twin_q"]:
			twin_q_t = model.get_twin_q_values(
				model_out_t, train_batch[SampleBatch.ACTIONS])

		# Q-values for current policy in given current state.
		q_t_det_policy = model.get_q_values(model_out_t, policy_t)
		if policy.config["twin_q"]:
			twin_q_t_det_policy = model.get_twin_q_values(
				model_out_t, policy_t)
			q_t_det_policy = tf.reduce_min(
				(q_t_det_policy, twin_q_t_det_policy), axis=0)

		# target q network evaluation
		q_tp1 = policy.target_model.get_q_values(target_model_out_tp1,
												 policy_tp1)
		if policy.config["twin_q"]:
			twin_q_tp1 = policy.target_model.get_twin_q_values(
				target_model_out_tp1, policy_tp1)
			# Take min over both twin-NNs.
			q_tp1 = tf.reduce_min((q_tp1, twin_q_tp1), axis=0)

		q_t_selected = tf.squeeze(q_t, axis=len(q_t.shape) - 1)
		if policy.config["twin_q"]:
			twin_q_t_selected = tf.squeeze(twin_q_t, axis=len(q_t.shape) - 1)
		q_tp1 -= model.alpha * log_pis_tp1

		q_tp1_best = tf.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
		q_tp1_best_masked = (1.0 - tf.cast(train_batch[SampleBatch.DONES],
										   tf.float32)) * q_tp1_best

	# Compute RHS of bellman equation for the Q-loss (critic(s)).
	q_t_selected_target = tf.stop_gradient(
		train_batch[SampleBatch.REWARDS] +
		policy.config["gamma"]**policy.config["n_step"] * q_tp1_best_masked)

	# Compute the TD-error (potentially clipped).
	base_td_error = tf.math.abs(q_t_selected - q_t_selected_target)
	if policy.config["twin_q"]:
		twin_td_error = tf.math.abs(twin_q_t_selected - q_t_selected_target)
		td_error = 0.5 * (base_td_error + twin_td_error)
	else:
		td_error = base_td_error

	# Calculate one or two critic losses (2 in the twin_q case).
	prio_weights = tf.cast(train_batch[PRIO_WEIGHTS], tf.float32)
	critic_loss = [tf.reduce_mean(prio_weights * huber_loss(base_td_error))]
	if policy.config["twin_q"]:
		critic_loss.append(
			tf.reduce_mean(prio_weights * huber_loss(twin_td_error)))

	# Alpha- and actor losses.
	# Note: In the papers, alpha is used directly, here we take the log.
	# Discrete case: Multiply the action probs as weights with the original
	# loss terms (no expectations needed).
	if model.discrete:
		alpha_loss = tf.reduce_mean(
			prio_weights * tf.reduce_sum(
				tf.multiply(
					tf.stop_gradient(policy_t), -model.log_alpha *
					tf.stop_gradient(log_pis_t + model.target_entropy)),
				axis=-1))
		actor_loss = tf.reduce_mean(
			prio_weights * tf.reduce_sum(
				tf.multiply(
					# NOTE: No stop_grad around policy output here
					# (compare with q_t_det_policy for continuous case).
					policy_t,
					model.alpha * log_pis_t - tf.stop_gradient(q_t)),
				axis=-1))
	else:
		alpha_loss = -tf.reduce_mean(
			prio_weights * 
			model.log_alpha *
			tf.stop_gradient(log_pis_t + model.target_entropy))
		actor_loss = tf.reduce_mean(prio_weights * (model.alpha * log_pis_t - q_t_det_policy))

	# Save for stats function.
	policy.policy_t = policy_t
	policy.q_t = q_t
	policy.td_error = td_error
	policy.actor_loss = actor_loss
	policy.critic_loss = critic_loss
	policy.alpha_loss = alpha_loss
	policy.alpha_value = model.alpha
	policy.target_entropy = model.target_entropy

	# In a custom apply op we handle the losses separately, but return them
	# combined in one loss here.
	return actor_loss + tf.math.add_n(critic_loss) + alpha_loss

