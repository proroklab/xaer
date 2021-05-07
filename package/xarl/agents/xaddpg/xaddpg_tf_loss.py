from ray.rllib.agents.ddpg.ddpg_tf_policy import *

def xaddpg_actor_critic_loss(policy, model, _, train_batch):
	twin_q = policy.config["twin_q"]
	gamma = policy.config["gamma"]
	n_step = policy.config["n_step"]
	use_huber = policy.config["use_huber"]
	huber_threshold = policy.config["huber_threshold"]
	l2_reg = policy.config["l2_reg"]

	input_dict = {
		"obs": train_batch[SampleBatch.CUR_OBS],
		"is_training": True,
	}
	input_dict_next = {
		"obs": train_batch[SampleBatch.NEXT_OBS],
		"is_training": True,
	}

	model_out_t, _ = model(input_dict, [], None)
	model_out_tp1, _ = model(input_dict_next, [], None)
	target_model_out_tp1, _ = policy.target_model(input_dict_next, [], None)

	policy.target_q_func_vars = policy.target_model.variables()

	# Policy network evaluation.
	policy_t = model.get_policy_output(model_out_t)
	policy_tp1 = \
		policy.target_model.get_policy_output(target_model_out_tp1)

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

	# Q-values for current policy (no noise) in given current state
	q_t_det_policy = model.get_q_values(model_out_t, policy_t)

	if twin_q:
		twin_q_t = model.get_twin_q_values(model_out_t,
										   train_batch[SampleBatch.ACTIONS])

	# Target q-net(s) evaluation.
	q_tp1 = policy.target_model.get_q_values(target_model_out_tp1,
											 policy_tp1_smoothed)

	if twin_q:
		twin_q_tp1 = policy.target_model.get_twin_q_values(
			target_model_out_tp1, policy_tp1_smoothed)

	q_t_selected = tf.squeeze(q_t, axis=len(q_t.shape) - 1)
	if twin_q:
		twin_q_t_selected = tf.squeeze(twin_q_t, axis=len(q_t.shape) - 1)
		q_tp1 = tf.minimum(q_tp1, twin_q_tp1)

	q_tp1_best = tf.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
	q_tp1_best_masked = \
		(1.0 - tf.cast(train_batch[SampleBatch.DONES], tf.float32)) * \
		q_tp1_best

	# Compute RHS of bellman equation.
	q_t_selected_target = tf.stop_gradient(train_batch[SampleBatch.REWARDS] +
										   gamma**n_step * q_tp1_best_masked)

	# Compute the error (potentially clipped).
	if twin_q:
		td_error = q_t_selected - q_t_selected_target
		twin_td_error = twin_q_t_selected - q_t_selected_target
		if use_huber:
			errors = huber_loss(td_error, huber_threshold) + \
				huber_loss(twin_td_error, huber_threshold)
		else:
			errors = 0.5 * tf.math.square(td_error) + \
					 0.5 * tf.math.square(twin_td_error)
	else:
		td_error = q_t_selected - q_t_selected_target
		if use_huber:
			errors = huber_loss(td_error, huber_threshold)
		else:
			errors = 0.5 * tf.math.square(td_error)

	prio_weights = tf.cast(train_batch[PRIO_WEIGHTS], tf.float32)
	critic_loss = tf.reduce_mean(prio_weights * errors)
	actor_loss = -tf.reduce_mean(prio_weights * q_t_det_policy)

	# Add l2-regularization if required.
	if l2_reg is not None:
		for var in policy.model.policy_variables():
			if "bias" not in var.name:
				actor_loss += (l2_reg * tf.nn.l2_loss(var))
		for var in policy.model.q_variables():
			if "bias" not in var.name:
				critic_loss += (l2_reg * tf.nn.l2_loss(var))

	# Model self-supervised losses.
	if policy.config["use_state_preprocessor"]:
		# Expand input_dict in case custom_loss' need them.
		input_dict[SampleBatch.ACTIONS] = train_batch[SampleBatch.ACTIONS]
		input_dict[SampleBatch.REWARDS] = train_batch[SampleBatch.REWARDS]
		input_dict[SampleBatch.DONES] = train_batch[SampleBatch.DONES]
		input_dict[SampleBatch.NEXT_OBS] = train_batch[SampleBatch.NEXT_OBS]
		if log_once("ddpg_custom_loss"):
			logger.warning(
				"You are using a state-preprocessor with DDPG and "
				"therefore, `custom_loss` will be called on your Model! "
				"Please be aware that DDPG now uses the ModelV2 API, which "
				"merges all previously separate sub-models (policy_model, "
				"q_model, and twin_q_model) into one ModelV2, on which "
				"`custom_loss` is called, passing it "
				"[actor_loss, critic_loss] as 1st argument. "
				"You may have to change your custom loss function to handle "
				"this.")
		[actor_loss, critic_loss] = model.custom_loss(
			[actor_loss, critic_loss], input_dict)

	# Store values for stats function.
	policy.actor_loss = actor_loss
	policy.critic_loss = critic_loss
	policy.td_error = td_error
	policy.q_t = q_t

	# Return one loss value (even though we treat them separately in our
	# 2 optimizers: actor and critic).
	return policy.critic_loss + policy.actor_loss
