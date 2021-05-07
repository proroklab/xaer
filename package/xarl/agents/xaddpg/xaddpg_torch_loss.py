from ray.rllib.agents.ddpg.ddpg_torch_policy import *

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

	# Policy network evaluation.
	# prev_update_ops = set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS))
	policy_t = model.get_policy_output(model_out_t)
	# policy_batchnorm_update_ops = list(
	#	set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS)) - prev_update_ops)

	policy_tp1 = \
		policy.target_model.get_policy_output(target_model_out_tp1)

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

	# Q-values for current policy (no noise) in given current state
	q_t_det_policy = model.get_q_values(model_out_t, policy_t)

	actor_loss = -torch.mean(train_batch[PRIO_WEIGHTS] * q_t_det_policy)

	if twin_q:
		twin_q_t = model.get_twin_q_values(model_out_t,
										   train_batch[SampleBatch.ACTIONS])
	# q_batchnorm_update_ops = list(
	#	 set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS)) - prev_update_ops)

	# Target q-net(s) evaluation.
	q_tp1 = policy.target_model.get_q_values(target_model_out_tp1,
											 policy_tp1_smoothed)

	if twin_q:
		twin_q_tp1 = policy.target_model.get_twin_q_values(
			target_model_out_tp1, policy_tp1_smoothed)

	q_t_selected = torch.squeeze(q_t, axis=len(q_t.shape) - 1)
	if twin_q:
		twin_q_t_selected = torch.squeeze(twin_q_t, axis=len(q_t.shape) - 1)
		q_tp1 = torch.min(q_tp1, twin_q_tp1)

	q_tp1_best = torch.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
	q_tp1_best_masked = \
		(1.0 - train_batch[SampleBatch.DONES].float()) * \
		q_tp1_best

	# Compute RHS of bellman equation.
	q_t_selected_target = (train_batch[SampleBatch.REWARDS] +
						   gamma**n_step * q_tp1_best_masked).detach()

	# Compute the error (potentially clipped).
	if twin_q:
		td_error = q_t_selected - q_t_selected_target
		twin_td_error = twin_q_t_selected - q_t_selected_target
		if use_huber:
			errors = huber_loss(td_error, huber_threshold) \
				+ huber_loss(twin_td_error, huber_threshold)
		else:
			errors = 0.5 * \
				(torch.pow(td_error, 2.0) + torch.pow(twin_td_error, 2.0))
	else:
		td_error = q_t_selected - q_t_selected_target
		if use_huber:
			errors = huber_loss(td_error, huber_threshold)
		else:
			errors = 0.5 * torch.pow(td_error, 2.0)

	critic_loss = torch.mean(train_batch[PRIO_WEIGHTS] * errors)

	# Add l2-regularization if required.
	if l2_reg is not None:
		for name, var in policy.model.policy_variables(as_dict=True).items():
			if "bias" not in name:
				actor_loss += (l2_reg * l2_loss(var))
		for name, var in policy.model.q_variables(as_dict=True).items():
			if "bias" not in name:
				critic_loss += (l2_reg * l2_loss(var))

	# Model self-supervised losses.
	if policy.config["use_state_preprocessor"]:
		# Expand input_dict in case custom_loss' need them.
		input_dict[SampleBatch.ACTIONS] = train_batch[SampleBatch.ACTIONS]
		input_dict[SampleBatch.REWARDS] = train_batch[SampleBatch.REWARDS]
		input_dict[SampleBatch.DONES] = train_batch[SampleBatch.DONES]
		input_dict[SampleBatch.NEXT_OBS] = train_batch[SampleBatch.NEXT_OBS]
		[actor_loss, critic_loss] = model.custom_loss(
			[actor_loss, critic_loss], input_dict)

	# Store values for stats function.
	policy.actor_loss = actor_loss
	policy.critic_loss = critic_loss
	policy.td_error = td_error
	policy.q_t = q_t

	# Return two loss terms (corresponding to the two optimizers, we create).
	return policy.actor_loss, policy.critic_loss

