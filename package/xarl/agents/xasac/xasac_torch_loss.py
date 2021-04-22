"""
PyTorch policy class used for SAC.
"""
from ray.rllib.agents.sac.sac_torch_policy import *
from ray.rllib.agents.sac.sac_torch_policy import _get_dist_class

def xasac_actor_critic_loss(policy, model, dist_class, train_batch):
	# Should be True only for debugging purposes (e.g. test cases)!
	deterministic = policy.config["_deterministic_loss"]

	model_out_t, _ = model({
		"obs": train_batch[SampleBatch.CUR_OBS],
		"is_training": True,
	}, [], None)

	model_out_tp1, _ = model({
		"obs": train_batch[SampleBatch.NEXT_OBS],
		"is_training": True,
	}, [], None)

	target_model_out_tp1, _ = policy.target_model({
		"obs": train_batch[SampleBatch.NEXT_OBS],
		"is_training": True,
	}, [], None)

	alpha = torch.exp(model.log_alpha)

	# Discrete case.
	if model.discrete:
		# Get all action probs directly from pi and form their logp.
		log_pis_t = F.log_softmax(model.get_policy_output(model_out_t), dim=-1)
		policy_t = torch.exp(log_pis_t)
		log_pis_tp1 = F.log_softmax(model.get_policy_output(model_out_tp1), -1)
		policy_tp1 = torch.exp(log_pis_tp1)
		# Q-values.
		q_t = model.get_q_values(model_out_t)
		# Target Q-values.
		q_tp1 = policy.target_model.get_q_values(target_model_out_tp1)
		if policy.config["twin_q"]:
			twin_q_t = model.get_twin_q_values(model_out_t)
			twin_q_tp1 = policy.target_model.get_twin_q_values(
				target_model_out_tp1)
			q_tp1 = torch.min(q_tp1, twin_q_tp1)
		q_tp1 -= alpha * log_pis_tp1

		# Actually selected Q-values (from the actions batch).
		one_hot = F.one_hot(
			train_batch[SampleBatch.ACTIONS].long(),
			num_classes=q_t.size()[-1])
		q_t_selected = torch.sum(q_t * one_hot, dim=-1)
		if policy.config["twin_q"]:
			twin_q_t_selected = torch.sum(twin_q_t * one_hot, dim=-1)
		# Discrete case: "Best" means weighted by the policy (prob) outputs.
		q_tp1_best = torch.sum(torch.mul(policy_tp1, q_tp1), dim=-1)
		q_tp1_best_masked = \
			(1.0 - train_batch[SampleBatch.DONES].float()) * \
			q_tp1_best
	# Continuous actions case.
	else:
		# Sample single actions from distribution.
		action_dist_class = _get_dist_class(policy.config, policy.action_space)
		action_dist_t = action_dist_class(
			model.get_policy_output(model_out_t), policy.model)
		policy_t = action_dist_t.sample() if not deterministic else \
			action_dist_t.deterministic_sample()
		log_pis_t = torch.unsqueeze(action_dist_t.logp(policy_t), -1)
		action_dist_tp1 = action_dist_class(
			model.get_policy_output(model_out_tp1), policy.model)
		policy_tp1 = action_dist_tp1.sample() if not deterministic else \
			action_dist_tp1.deterministic_sample()
		log_pis_tp1 = torch.unsqueeze(action_dist_tp1.logp(policy_tp1), -1)

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
			q_t_det_policy = torch.min(q_t_det_policy, twin_q_t_det_policy)

		# Target q network evaluation.
		q_tp1 = policy.target_model.get_q_values(target_model_out_tp1,
												 policy_tp1)
		if policy.config["twin_q"]:
			twin_q_tp1 = policy.target_model.get_twin_q_values(
				target_model_out_tp1, policy_tp1)
			# Take min over both twin-NNs.
			q_tp1 = torch.min(q_tp1, twin_q_tp1)

		q_t_selected = torch.squeeze(q_t, dim=-1)
		if policy.config["twin_q"]:
			twin_q_t_selected = torch.squeeze(twin_q_t, dim=-1)
		q_tp1 -= alpha * log_pis_tp1

		q_tp1_best = torch.squeeze(input=q_tp1, dim=-1)
		q_tp1_best_masked = (1.0 - train_batch[SampleBatch.DONES].float()) * \
			q_tp1_best

	# compute RHS of bellman equation
	q_t_selected_target = (
		train_batch[SampleBatch.REWARDS] +
		(policy.config["gamma"]**policy.config["n_step"]) * q_tp1_best_masked
	).detach()

	# Compute the TD-error (potentially clipped).
	base_td_error = torch.abs(q_t_selected - q_t_selected_target)
	if policy.config["twin_q"]:
		twin_td_error = torch.abs(twin_q_t_selected - q_t_selected_target)
		td_error = 0.5 * (base_td_error + twin_td_error)
	else:
		td_error = base_td_error

	critic_loss = [
		torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(base_td_error))
	]
	if policy.config["twin_q"]:
		critic_loss.append(
			torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(twin_td_error)))

	# Alpha- and actor losses.
	# Note: In the papers, alpha is used directly, here we take the log.
	# Discrete case: Multiply the action probs as weights with the original
	# loss terms (no expectations needed).
	if model.discrete:
		weighted_log_alpha_loss = policy_t.detach() * (
			-model.log_alpha * (log_pis_t + model.target_entropy).detach())
		# Sum up weighted terms and mean over all batch items.
		alpha_loss = torch.mean(train_batch[PRIO_WEIGHTS] * torch.sum(weighted_log_alpha_loss, dim=-1))
		# Actor loss.
		actor_loss = torch.mean(
			train_batch[PRIO_WEIGHTS] * torch.sum(
				torch.mul(
					# NOTE: No stop_grad around policy output here
					# (compare with q_t_det_policy for continuous case).
					policy_t,
					alpha.detach() * log_pis_t - q_t.detach()),
				dim=-1))
	else:
		alpha_loss = -torch.mean(train_batch[PRIO_WEIGHTS] * model.log_alpha * (log_pis_t + model.target_entropy).detach())
		# Note: Do not detach q_t_det_policy here b/c is depends partly
		# on the policy vars (policy sample pushed through Q-net).
		# However, we must make sure `actor_loss` is not used to update
		# the Q-net(s)' variables.
		actor_loss = torch.mean(train_batch[PRIO_WEIGHTS] * (alpha.detach() * log_pis_t - q_t_det_policy))

	# Save for stats function.
	policy.q_t = q_t
	policy.policy_t = policy_t
	policy.log_pis_t = log_pis_t

	# Store td-error in model, such that for multi-GPU, we do not override
	# them during the parallel loss phase. TD-error tensor in final stats
	# can then be concatenated and retrieved for each individual batch item.
	model.td_error = td_error

	policy.actor_loss = actor_loss
	policy.critic_loss = critic_loss
	policy.alpha_loss = alpha_loss
	policy.log_alpha_value = model.log_alpha
	policy.alpha_value = alpha
	policy.target_entropy = model.target_entropy

	# Return all loss terms corresponding to our optimizers.
	return tuple([policy.actor_loss] + policy.critic_loss +
				 [policy.alpha_loss])
