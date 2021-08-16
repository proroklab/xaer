from ray.rllib.agents.sac.sac_tf_model import *
from xarl.models.appo import TFAdaptiveMultiHeadNet as APPOTFAdaptiveMultiHeadNet
from xarl.models.adaptive_model_wrapper import get_tf_heads_model, get_heads_input, tf


class TFAdaptiveMultiHeadNet(SACTFModel):
	"""Extension of the standard TFModelV2 for SAC.

	To customize, do one of the following:
	- sub-class SACTFModel and override one or more of its methods.
	- Use SAC's `Q_model` and `policy_model` keys to tweak the default model
	  behaviors (e.g. fcnet_hiddens, conv_filters, etc..).
	- Use SAC's `Q_model->custom_model` and `policy_model->custom_model` keys
	  to specify your own custom Q-model(s) and policy-models, which will be
	  created within this SACTFModel (see `build_policy_model` and
	  `build_q_model`.

	Note: It is not recommended to override the `forward` method for SAC. This
	would lead to shared weights (between policy and Q-nets), which will then
	not be optimized by either of the critic- or actor-optimizers!

	Data flow:
		`obs` -> forward() (should stay a noop method!) -> `model_out`
		`model_out` -> get_policy_output() -> pi(actions|obs)
		`model_out`, `actions` -> get_q_values() -> Q(s, a)
		`model_out`, `actions` -> get_twin_q_values() -> Q_twin(s, a)
	"""

	def build_policy_model(self, obs_space, num_outputs, policy_model_config,
						   name):
		"""Builds the policy model used by this SAC.

		Override this method in a sub-class of SACTFModel to implement your
		own policy net. Alternatively, simply set `custom_model` within the
		top level SAC `policy_model` config key to make this default
		implementation of `build_policy_model` use your custom policy network.

		Returns:
			TFModelV2: The TFModelV2 policy sub-model.
		"""

		model = APPOTFAdaptiveMultiHeadNet(
			obs_space,
			self.action_space,
			num_outputs,
			policy_model_config,
			name=name)
		return model

	def build_q_model(self, obs_space, action_space, num_outputs,
					  q_model_config, name):
		"""Builds one of the (twin) Q-nets used by this SAC.

		Override this method in a sub-class of SACTFModel to implement your
		own Q-nets. Alternatively, simply set `custom_model` within the
		top level SAC `Q_model` config key to make this default implementation
		of `build_q_model` use your custom Q-nets.

		Returns:
			TFModelV2: The TFModelV2 Q-net sub-model.
		"""
		inputs, last_layer = get_tf_heads_model(obs_space)
		self.head_q_model = tf.keras.Model(inputs, last_layer)
		# self.register_variables(self.head_q_model.variables)

		self.concat_obs_and_actions = False
		if self.discrete:
			input_space = obs_space
		else:
			orig_space = getattr(obs_space, "original_space", obs_space)
			input_space = Box(
				float("-inf"),
				float("inf"),
				shape=(last_layer.shape[1] + action_space.shape[0], ))
			self.concat_obs_and_actions = True

		model = ModelCatalog.get_model_v2(
			input_space,
			action_space,
			num_outputs,
			q_model_config,
			framework="tf",
			name=name)
		return model

	def _get_q_value(self, model_out, actions, net):
		model_out = self.head_q_model(get_heads_input({"obs": model_out}))
		# Model outs may come as original Tuple/Dict observations, concat them
		# here if this is the case.
		if isinstance(net.obs_space, Box):
			if isinstance(model_out, (list, tuple)):
				model_out = tf.concat(model_out, axis=-1)
			elif isinstance(model_out, dict):
				model_out = tf.concat(list(model_out.values()), axis=-1)
		elif isinstance(model_out, dict):
			model_out = list(model_out.values())

		# Continuous case -> concat actions to model_out.
		if actions is not None:
			if self.concat_obs_and_actions:
				input_dict = {"obs": tf.concat([model_out, actions], axis=-1)}
			else:
				input_dict = {"obs": force_list(model_out) + [actions]}
		# Discrete case -> return q-vals for all actions.
		else:
			input_dict = {"obs": model_out}
		# Switch on training mode (when getting Q-values, we are usually in
		# training).
		input_dict["is_training"] = True

		out, _ = net(input_dict, [], None)
		return out

	def get_policy_output(self, model_out: TensorType) -> TensorType:
		"""Returns policy outputs, given the output of self.__call__().

		For continuous action spaces, these will be the mean/stddev
		distribution inputs for the (SquashedGaussian) action distribution.
		For discrete action spaces, these will be the logits for a categorical
		distribution.

		Args:
			model_out (TensorType): Feature outputs from the model layers
				(result of doing `self.__call__(obs)`).

		Returns:
			TensorType: Distribution inputs for sampling actions.
		"""
		# Model outs may come as original Tuple observations, concat them
		# here if this is the case.
		out, _ = self.action_model({"obs": model_out}, [], None)
		return out
