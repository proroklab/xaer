from ray.rllib.models.tf.fcnet import FullyConnectedNetwork as FCNet
from xarl.models.adaptive_model_wrapper import get_tf_heads_model, get_heads_input, tf
from ray.rllib.models.modelv2 import restore_original_dimensions
import numpy as np

class TFAdaptiveMultiHeadAPPO(FCNet):
	def __init__(self, obs_space, action_space, num_outputs, model_config, name, **args):
		self._original_obs_space = obs_space
		inputs, layer = get_tf_heads_model(obs_space, num_outputs)
		super().__init__(
			obs_space=np.zeros(layer.shape[1]), 
			action_space=action_space, 
			num_outputs=layer.shape[1], 
			model_config=model_config, 
			name=name, 
			**args
		)
		self.heads_model = tf.keras.Model(inputs, layer)
		self.register_variables(self.heads_model.variables)

	def forward(self, input_dict, state, seq_lens):
		input_dict["obs"] = restore_original_dimensions(input_dict["obs"], self._original_obs_space, 'tf')
		model_out = self.heads_model(get_heads_input(input_dict))
		model_out, self._value_out = self.base_model(model_out)
		return model_out, state
