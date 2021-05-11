from ray.rllib.models.tf.tf_action_dist import *

class FixedDiagGaussian(DiagGaussian):
	def __init__(self, inputs: List[TensorType], model: ModelV2):
		super().__init__(inputs, model)
		# Clip `scale` values (coming from NN) to reasonable values.
		self.log_std = tf.clip_by_value(self.log_std, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT)
		self.std = tf.exp(self.log_std)
