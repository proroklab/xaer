from ray.rllib.models.torch.torch_action_dist import *

class FixedTorchDiagGaussian(TorchDiagGaussian):
	@override(ActionDistribution)
	def __init__(self, inputs: List[TensorType], model: TorchModelV2):
		super().__init__(inputs, model)
		mean, log_std = torch.chunk(self.inputs, 2, dim=1)
		log_std = torch.clamp(log_std, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT)
		self.dist = torch.distributions.normal.Normal(mean, torch.exp(log_std))
