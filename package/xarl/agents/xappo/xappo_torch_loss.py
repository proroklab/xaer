"""
PyTorch policy class used for APPO.

Adapted from VTraceTFPolicy to use the PPO surrogate loss.
Keep in sync with changes to VTraceTFPolicy.
"""
from ray.rllib.agents.ppo.appo_torch_policy import *
from ray.rllib.agents.dqn.dqn_tf_policy import PRIO_WEIGHTS

def xappo_surrogate_loss(policy, model, dist_class, train_batch):
	"""Constructs the loss for APPO.

	With IS modifications and V-trace for Advantage Estimation.

	Args:
		policy (Policy): The Policy to calculate the loss for.
		model (ModelV2): The Model to calculate the loss for.
		dist_class (Type[ActionDistribution]: The action distr. class.
		train_batch (SampleBatch): The training data.

	Returns:
		Union[TensorType, List[TensorType]]: A single loss tensor or a list
			of loss tensors.
	"""
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

	def _make_time_major(*args, **kw):
		return make_time_major(policy, train_batch.get("seq_lens"), *args,
							   **kw)

	actions = train_batch[SampleBatch.ACTIONS]
	dones = train_batch[SampleBatch.DONES]
	rewards = train_batch[SampleBatch.REWARDS]
	behaviour_logits = train_batch[SampleBatch.ACTION_DIST_INPUTS]

	target_model_out, _ = policy.target_model.from_batch(train_batch)

	prev_action_dist = dist_class(behaviour_logits, policy.model)
	values = policy.model.value_function()
	values_time_major = _make_time_major(values)

	policy.model_vars = policy.model.variables()
	policy.target_model_vars = policy.target_model.variables()

	if policy.is_recurrent():
		max_seq_len = torch.max(train_batch["seq_lens"]) - 1
		mask = sequence_mask(train_batch["seq_lens"], max_seq_len)
		mask = torch.reshape(mask, [-1])
		num_valid = torch.sum(mask)

		def reduce_mean_valid(t):
			return torch.sum(t * mask) / num_valid

	else:
		reduce_mean_valid = torch.mean

	weights = _make_time_major(train_batch[PRIO_WEIGHTS], drop_last=policy.config["vtrace"])

	if policy.config["vtrace"]:
		logger.debug("Using V-Trace surrogate loss (vtrace=True)")

		old_policy_behaviour_logits = target_model_out.detach()
		old_policy_action_dist = dist_class(old_policy_behaviour_logits, model)

		if isinstance(output_hidden_shape, (list, tuple, np.ndarray)):
			unpacked_behaviour_logits = torch.split(
				behaviour_logits, list(output_hidden_shape), dim=1)
			unpacked_old_policy_behaviour_logits = torch.split(
				old_policy_behaviour_logits, list(output_hidden_shape), dim=1)
		else:
			unpacked_behaviour_logits = torch.chunk(
				behaviour_logits, output_hidden_shape, dim=1)
			unpacked_old_policy_behaviour_logits = torch.chunk(
				old_policy_behaviour_logits, output_hidden_shape, dim=1)

		# Prepare actions for loss.
		loss_actions = actions if is_multidiscrete else torch.unsqueeze(
			actions, dim=1)

		# Prepare KL for loss.
		action_kl = _make_time_major(
			old_policy_action_dist.kl(action_dist), drop_last=True)

		# Compute vtrace on the CPU for better perf.
		vtrace_returns = vtrace.multi_from_logits(
			behaviour_policy_logits=_make_time_major(
				unpacked_behaviour_logits, drop_last=True),
			target_policy_logits=_make_time_major(
				unpacked_old_policy_behaviour_logits, drop_last=True),
			actions=torch.unbind(
				_make_time_major(loss_actions, drop_last=True), dim=2),
			discounts=(1.0 - _make_time_major(dones, drop_last=True).float()) *
			policy.config["gamma"],
			rewards=_make_time_major(rewards, drop_last=True),
			values=values_time_major[:-1],  # drop-last=True
			bootstrap_value=values_time_major[-1],
			dist_class=TorchCategorical if is_multidiscrete else dist_class,
			model=model,
			clip_rho_threshold=policy.config["vtrace_clip_rho_threshold"],
			clip_pg_rho_threshold=policy.config[
				"vtrace_clip_pg_rho_threshold"])

		actions_logp = _make_time_major(
			action_dist.logp(actions), drop_last=True)
		prev_actions_logp = _make_time_major(
			prev_action_dist.logp(actions), drop_last=True)
		old_policy_actions_logp = _make_time_major(
			old_policy_action_dist.logp(actions), drop_last=True)
		is_ratio = torch.clamp(
			torch.exp(prev_actions_logp - old_policy_actions_logp), 0.0, 2.0)
		logp_ratio = is_ratio * torch.exp(actions_logp - prev_actions_logp)
		policy._is_ratio = is_ratio

		advantages = vtrace_returns.pg_advantages.to(policy.device)
		surrogate_loss = torch.min(
			advantages * logp_ratio,
			advantages *
			torch.clamp(logp_ratio, 1 - policy.config["clip_param"],
						1 + policy.config["clip_param"]))

		if PRIO_WEIGHTS in train_batch:
			surrogate_loss *= weights
			action_kl *= weights
		mean_kl = reduce_mean_valid(action_kl)
		mean_policy_loss = -reduce_mean_valid(surrogate_loss)

		# The value function loss.
		value_targets = vtrace_returns.vs.to(policy.device)
		delta = values_time_major[:-1] - value_targets
		if PRIO_WEIGHTS in train_batch:
			delta *= weights
		mean_vf_loss = 0.5 * reduce_mean_valid(torch.pow(delta, 2.0))

		# The entropy loss.
		entropy = _make_time_major(action_dist.entropy(), drop_last=True)
		if PRIO_WEIGHTS in train_batch:
			entropy *= weights
		mean_entropy = reduce_mean_valid(entropy)

	else:
		logger.debug("Using PPO surrogate loss (vtrace=False)")

		# Prepare KL for Loss
		action_kl = _make_time_major(prev_action_dist.kl(action_dist))

		actions_logp = _make_time_major(action_dist.logp(actions))
		prev_actions_logp = _make_time_major(prev_action_dist.logp(actions))
		logp_ratio = torch.exp(actions_logp - prev_actions_logp)

		advantages = _make_time_major(train_batch[Postprocessing.ADVANTAGES])
		surrogate_loss = torch.min(
			advantages * logp_ratio,
			advantages *
			torch.clamp(logp_ratio, 1 - policy.config["clip_param"],
						1 + policy.config["clip_param"]))

		if PRIO_WEIGHTS in train_batch:
			surrogate_loss *= weights
			action_kl *= weights
		mean_kl = reduce_mean_valid(action_kl)
		mean_policy_loss = -reduce_mean_valid(surrogate_loss)

		# The value function loss.
		value_targets = _make_time_major(train_batch[Postprocessing.VALUE_TARGETS])
		delta = values_time_major - value_targets
		if PRIO_WEIGHTS in train_batch:
			delta *= weights
		mean_vf_loss = 0.5 * reduce_mean_valid(torch.pow(delta, 2.0))

		# The entropy loss.
		entropy = _make_time_major(action_dist.entropy())
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

	return total_loss
