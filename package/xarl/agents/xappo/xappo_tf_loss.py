"""
TensorFlow policy class used for APPO.

Adapted from VTraceTFPolicy to use the PPO surrogate loss.
Keep in sync with changes to VTraceTFPolicy.
"""

from ray.rllib.agents.ppo.appo_tf_policy import *
from ray.rllib.agents.ppo.appo_tf_policy import _make_time_major
from ray.rllib.agents.dqn.dqn_tf_policy import PRIO_WEIGHTS

def xappo_surrogate_loss(policy, model, dist_class, train_batch):
	model_out, _ = model.from_batch(train_batch)
	action_dist = dist_class(model_out, model)

	if isinstance(policy.action_space, gym.spaces.Discrete):
		is_multidiscrete = False
		output_hidden_shape = [policy.action_space.n]
	elif isinstance(policy.action_space,
					gym.spaces.multi_discrete.MultiDiscrete):
		is_multidiscrete = True
		output_hidden_shape = policy.action_space.nvec.astype(np.int32)
	else:
		is_multidiscrete = False
		output_hidden_shape = 1

	# TODO: (sven) deprecate this when trajectory view API gets activated.
	def make_time_major(*args, **kw):
		return _make_time_major(policy, train_batch.get("seq_lens"), *args, **kw)

	actions = train_batch[SampleBatch.ACTIONS]
	dones = train_batch[SampleBatch.DONES]
	rewards = train_batch[SampleBatch.REWARDS]
	behaviour_logits = train_batch[SampleBatch.ACTION_DIST_INPUTS]

	target_model_out, _ = policy.target_model.from_batch(train_batch)
	prev_action_dist = dist_class(behaviour_logits, policy.model)
	values = policy.model.value_function()
	values_time_major = make_time_major(values)

	policy.model_vars = policy.model.variables()
	policy.target_model_vars = policy.target_model.variables()

	if PRIO_WEIGHTS in train_batch:
		weights = make_time_major(train_batch[PRIO_WEIGHTS], drop_last=policy.config["vtrace"])

	if policy.is_recurrent():
		max_seq_len = tf.reduce_max(train_batch["seq_lens"]) - 1
		mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
		mask = tf.reshape(mask, [-1])
		mask = make_time_major(mask, drop_last=policy.config["vtrace"])

		def reduce_mean_valid(t):
			return tf.reduce_mean(tf.boolean_mask(t, mask))

	else:
		reduce_mean_valid = tf.reduce_mean

	if policy.config["vtrace"]:
		logger.debug("Using V-Trace surrogate loss (vtrace=True)")

		# Prepare actions for loss.
		loss_actions = actions if is_multidiscrete else tf.expand_dims(
			actions, axis=1)

		old_policy_behaviour_logits = tf.stop_gradient(target_model_out)
		old_policy_action_dist = dist_class(old_policy_behaviour_logits, model)

		# Prepare KL for Loss
		mean_kl = make_time_major(
			old_policy_action_dist.multi_kl(action_dist), drop_last=True)

		unpacked_behaviour_logits = tf.split(
			behaviour_logits, output_hidden_shape, axis=1)
		unpacked_old_policy_behaviour_logits = tf.split(
			old_policy_behaviour_logits, output_hidden_shape, axis=1)

		# Compute vtrace on the CPU for better perf.
		with tf.device("/cpu:0"):
			vtrace_returns = vtrace.multi_from_logits(
				behaviour_policy_logits=make_time_major(
					unpacked_behaviour_logits, drop_last=True),
				target_policy_logits=make_time_major(
					unpacked_old_policy_behaviour_logits, drop_last=True),
				actions=tf.unstack(
					make_time_major(loss_actions, drop_last=True), axis=2),
				discounts=tf.cast(
					~make_time_major(tf.cast(dones, tf.bool), drop_last=True),
					tf.float32) * policy.config["gamma"],
				rewards=make_time_major(rewards, drop_last=True),
				values=values_time_major[:-1],  # drop-last=True
				bootstrap_value=values_time_major[-1],
				dist_class=Categorical if is_multidiscrete else dist_class,
				model=model,
				clip_rho_threshold=tf.cast(
					policy.config["vtrace_clip_rho_threshold"], tf.float32),
				clip_pg_rho_threshold=tf.cast(
					policy.config["vtrace_clip_pg_rho_threshold"], tf.float32),
			)

		actions_logp = make_time_major(
			action_dist.logp(actions), drop_last=True)
		prev_actions_logp = make_time_major(
			prev_action_dist.logp(actions), drop_last=True)
		old_policy_actions_logp = make_time_major(
			old_policy_action_dist.logp(actions), drop_last=True)

		is_ratio = tf.clip_by_value(
			tf.math.exp(prev_actions_logp - old_policy_actions_logp), 0.0, 2.0)
		logp_ratio = is_ratio * tf.exp(actions_logp - prev_actions_logp)
		policy._is_ratio = is_ratio

		advantages = vtrace_returns.pg_advantages
		surrogate_loss = tf.minimum(
			advantages * logp_ratio,
			advantages *
			tf.clip_by_value(logp_ratio, 1 - policy.config["clip_param"],
							 1 + policy.config["clip_param"]))

		action_kl = tf.reduce_mean(mean_kl, axis=0) if is_multidiscrete else mean_kl
		if PRIO_WEIGHTS in train_batch:
			surrogate_loss *= weights
			action_kl *= weights
		mean_kl = reduce_mean_valid(action_kl)
		mean_policy_loss = -reduce_mean_valid(surrogate_loss)

		# The value function loss.
		delta = values_time_major[:-1] - vtrace_returns.vs
		value_targets = vtrace_returns.vs
		if PRIO_WEIGHTS in train_batch:
			delta *= weights
		mean_vf_loss = 0.5 * reduce_mean_valid(tf.math.square(delta))

		# The entropy loss.
		actions_entropy = make_time_major(action_dist.multi_entropy(), drop_last=True)
		if PRIO_WEIGHTS in train_batch:
			actions_entropy *= weights
		mean_entropy = reduce_mean_valid(actions_entropy)

	else:
		logger.debug("Using PPO surrogate loss (vtrace=False)")

		# Prepare KL for Loss
		mean_kl = make_time_major(prev_action_dist.multi_kl(action_dist))

		logp_ratio = tf.math.exp(
			make_time_major(action_dist.logp(actions)) -
			make_time_major(prev_action_dist.logp(actions)))

		advantages = make_time_major(train_batch[Postprocessing.ADVANTAGES])
		surrogate_loss = tf.minimum(
			advantages * logp_ratio,
			advantages *
			tf.clip_by_value(logp_ratio, 1 - policy.config["clip_param"],
							 1 + policy.config["clip_param"]))

		action_kl = tf.reduce_mean(mean_kl, axis=0) if is_multidiscrete else mean_kl
		if PRIO_WEIGHTS in train_batch:
			surrogate_loss *= weights
			action_kl *= weights
		mean_kl = reduce_mean_valid(action_kl)
		mean_policy_loss = -reduce_mean_valid(surrogate_loss)

		# The value function loss.
		value_targets = make_time_major(
			train_batch[Postprocessing.VALUE_TARGETS])
		delta = values_time_major - value_targets
		if PRIO_WEIGHTS in train_batch:
			delta *= weights
		mean_vf_loss = 0.5 * reduce_mean_valid(tf.math.square(delta))

		# The entropy loss.
		entropy = make_time_major(action_dist.multi_entropy())
		if PRIO_WEIGHTS in train_batch:
			entropy *= weights
		mean_entropy = reduce_mean_valid(entropy)

	# The summed weighted loss
	total_loss = mean_policy_loss + \
		mean_vf_loss * policy.config["vf_loss_coeff"] - \
		mean_entropy * policy.config["entropy_coeff"]

	# Optional additional KL Loss
	if policy.config["use_kl_loss"]:
		total_loss += policy.kl_coeff * mean_kl

	policy._total_loss = total_loss
	policy._mean_policy_loss = mean_policy_loss
	policy._mean_kl = mean_kl
	policy._mean_vf_loss = mean_vf_loss
	policy._mean_entropy = mean_entropy
	policy._value_targets = value_targets

	# Store stats in policy for stats_fn.
	return total_loss
