"""
PyTorch policy class used for CQL.
"""
from ray.rllib.agents.cql.cql_torch_policy import *
from ray.rllib.agents.cql.cql_torch_policy import _get_dist_class
from ray.rllib.agents.dqn.dqn_tf_policy import PRIO_WEIGHTS

def cql_loss(policy, model, dist_class, train_batch):
	# print(policy.cur_iter)
	policy.cur_iter += 1
	# For best performance, turn deterministic off
	deterministic = policy.config["_deterministic_loss"]
	twin_q = policy.config["twin_q"]
	discount = policy.config["gamma"]
	action_low = model.action_space.low[0]
	action_high = model.action_space.high[0]

	# CQL Parameters
	bc_iters = policy.config["bc_iters"]
	cql_temp = policy.config["temperature"]
	num_actions = policy.config["num_actions"]
	min_q_weight = policy.config["min_q_weight"]
	use_lagrange = policy.config["lagrangian"]
	target_action_gap = policy.config["lagrangian_thresh"]

	obs = train_batch[SampleBatch.CUR_OBS]
	actions = train_batch[SampleBatch.ACTIONS]
	rewards = train_batch[SampleBatch.REWARDS]
	next_obs = train_batch[SampleBatch.NEXT_OBS]
	terminals = train_batch[SampleBatch.DONES]

	model_out_t, _ = model({
		"obs": obs,
		"is_training": True,
	}, [], None)

	model_out_tp1, _ = model({
		"obs": next_obs,
		"is_training": True,
	}, [], None)

	target_model_out_tp1, _ = policy.target_model({
		"obs": next_obs,
		"is_training": True,
	}, [], None)

	action_dist_class = _get_dist_class(policy.config, policy.action_space)
	action_dist_t = action_dist_class(
		model.get_policy_output(model_out_t), policy.model)
	policy_t = action_dist_t.sample() if not deterministic else \
		action_dist_t.deterministic_sample()
	log_pis_t = torch.unsqueeze(action_dist_t.logp(policy_t), -1)

	# Unlike original SAC, Alpha and Actor Loss are computed first.
	# Alpha Loss
	alpha_loss = -torch.mean(train_batch[PRIO_WEIGHTS] * model.log_alpha * (log_pis_t + model.target_entropy).detach())

	# Policy Loss (Either Behavior Clone Loss or SAC Loss)
	alpha = torch.exp(model.log_alpha)
	if policy.cur_iter >= bc_iters:
		min_q = model.get_q_values(model_out_t, policy_t)
		if twin_q:
			min_q = torch.min(min_q, model.get_twin_q_values(model_out_t, policy_t))
		actor_loss = torch.mean(train_batch[PRIO_WEIGHTS] * (alpha.detach() * log_pis_t - min_q))
	else:
		bc_logp = action_dist_t.logp(actions)
		actor_loss = torch.mean(train_batch[PRIO_WEIGHTS] * (alpha * log_pis_t - bc_logp))

	# Critic Loss (Standard SAC Critic L2 Loss + CQL Entropy Loss)
	# SAC Loss
	action_dist_tp1 = action_dist_class(model.get_policy_output(model_out_tp1), policy.model)
	policy_tp1 = action_dist_tp1.sample() if not deterministic else action_dist_tp1.deterministic_sample()

	# Q-values for the batched actions.
	q_t = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
	q_t = torch.squeeze(q_t, dim=-1)
	if twin_q:
		twin_q_t = model.get_twin_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
		twin_q_t = torch.squeeze(twin_q_t, dim=-1)

	# Target q network evaluation.
	q_tp1 = policy.target_model.get_q_values(target_model_out_tp1, policy_tp1)
	if twin_q:
		twin_q_tp1 = policy.target_model.get_twin_q_values(target_model_out_tp1, policy_tp1)
		# Take min over both twin-NNs.
		q_tp1 = torch.min(q_tp1, twin_q_tp1)
	q_tp1 = torch.squeeze(input=q_tp1, dim=-1)
	q_tp1 = (1.0 - terminals.float()) * q_tp1

	# compute RHS of bellman equation
	q_t_target = (rewards + (discount**policy.config["n_step"]) * q_tp1).detach()

	# Compute the TD-error (potentially clipped), for priority replay buffer
	base_td_error = torch.abs(q_t - q_t_target)
	if twin_q:
		twin_td_error = torch.abs(twin_q_t - q_t_target)
		td_error = 0.5 * (base_td_error + twin_td_error)
	else:
		td_error = base_td_error
	critic_loss = [nn.MSELoss()(train_batch[PRIO_WEIGHTS] * q_t, train_batch[PRIO_WEIGHTS] * q_t_target)]
	if twin_q:
		critic_loss.append(nn.MSELoss()(train_batch[PRIO_WEIGHTS] * twin_q_t, train_batch[PRIO_WEIGHTS] * q_t_target))

	# CQL Loss (We are using Entropy version of CQL (the best version))
	rand_actions = convert_to_torch_tensor(
		torch.FloatTensor(actions.shape[0] * num_actions, actions.shape[-1]).uniform_(action_low, action_high),
		policy.device
	)
	curr_actions, curr_logp = policy_actions_repeat(model, action_dist_class, obs, num_actions)
	next_actions, next_logp = policy_actions_repeat(model, action_dist_class, next_obs, num_actions)

	curr_logp = curr_logp.view(actions.shape[0], num_actions, 1)
	next_logp = next_logp.view(actions.shape[0], num_actions, 1)

	q1_rand = q_values_repeat(model, model_out_t, rand_actions)
	q1_curr_actions = q_values_repeat(model, model_out_t, curr_actions)
	q1_next_actions = q_values_repeat(model, model_out_t, next_actions)

	if twin_q:
		q2_rand = q_values_repeat(model, model_out_t, rand_actions, twin=True)
		q2_curr_actions = q_values_repeat(model, model_out_t, curr_actions, twin=True)
		q2_next_actions = q_values_repeat(model, model_out_t, next_actions, twin=True)

	random_density = np.log(0.5**curr_actions.shape[-1])
	cat_q1 = torch.cat([
		q1_rand - random_density, q1_next_actions - next_logp.detach(),
		q1_curr_actions - curr_logp.detach()
	], 1)
	if twin_q:
		cat_q2 = torch.cat([
			q2_rand - random_density, q2_next_actions - next_logp.detach(),
			q2_curr_actions - curr_logp.detach()
		], 1)

	min_qf1_loss = torch.mean(train_batch[PRIO_WEIGHTS] * torch.logsumexp(cat_q1 / cql_temp, dim=1)) * min_q_weight * cql_temp
	min_qf1_loss = min_qf1_loss - torch.mean(train_batch[PRIO_WEIGHTS] * q_t) * min_q_weight
	if twin_q:
		min_qf2_loss = torch.mean(train_batch[PRIO_WEIGHTS] * torch.logsumexp(cat_q2 / cql_temp, dim=1)) * min_q_weight * cql_temp
		min_qf2_loss = min_qf2_loss - torch.mean(train_batch[PRIO_WEIGHTS] * twin_q_t) * min_q_weight

	if use_lagrange:
		alpha_prime = torch.clamp(model.log_alpha_prime.exp(), min=0.0, max=1000000.0)[0]
		min_qf1_loss = alpha_prime * (min_qf1_loss - target_action_gap)
		if twin_q:
			min_qf2_loss = alpha_prime * (min_qf2_loss - target_action_gap)
			alpha_prime_loss = 0.5 * (-min_qf1_loss - min_qf2_loss)
		else:
			alpha_prime_loss = -min_qf1_loss

	cql_loss = [min_qf2_loss]
	if twin_q:
		cql_loss.append(min_qf2_loss)

	critic_loss[0] += min_qf1_loss
	if twin_q:
		critic_loss[1] += min_qf2_loss

	# Save for stats function.
	policy.q_t = q_t
	policy.policy_t = policy_t
	policy.log_pis_t = log_pis_t
	policy.td_error = td_error
	policy.actor_loss = actor_loss
	policy.critic_loss = critic_loss
	policy.alpha_loss = alpha_loss
	policy.log_alpha_value = model.log_alpha
	policy.alpha_value = alpha
	policy.target_entropy = model.target_entropy
	# CQL Stats
	policy.cql_loss = cql_loss
	if use_lagrange:
		policy.log_alpha_prime_value = model.log_alpha_prime[0]
		policy.alpha_prime_value = alpha_prime
		policy.alpha_prime_loss = alpha_prime_loss

	# Return all loss terms corresponding to our optimizers.
	if use_lagrange:
		return tuple([policy.actor_loss] + policy.critic_loss +
					 [policy.alpha_loss] + [policy.alpha_prime_loss])
	return tuple([policy.actor_loss] + policy.critic_loss +
				 [policy.alpha_loss])
