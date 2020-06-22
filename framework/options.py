# -*- coding: utf-8 -*-
# Decentralized Distributed Asynchronous
from types import SimpleNamespace

options = None
def build():
	global options
	options = {}
	options["max_timestep"] = 2**30 # "Max training time steps."
	options["timesteps_before_starting_training"] = 2**10 # "Number of initialization steps."
	options["shuffle_sequences"] = True # Whether to shuffle sequences in the batch when training (recommended).
# Environment
	options["env_type"] = "CarController" # "environment types: CarController or environments from https://gym.openai.com/envs"
# Gradient optimization parameters
	options["parameters_type"] = "float32" # "The type used to represent parameters: bfloat16, float32, float64"
	options["algorithm"] = "AC" # "algorithms: AC, TD3"
	options["network_configuration"] = "ExplicitlyArgumentative" # "neural network configurations: Base, Towers, HybridTowers, SA, OpenAISmall, OpenAILarge, Impala, ExplicitlyRelational, ExplicitlyArgumentative"
	options["network_has_internal_state"] = False # "Whether the network has an internal state to keep updated (eg. RNNs state)."
	options["optimizer"] = "Adam" # "gradient optimizer: PowerSign, AddSign, ElasticAverage, LazyAdam, Nadam, Adadelta, AdagradDA, Adagrad, Adam, Ftrl, GradientDescent, Momentum, ProximalAdagrad, ProximalGradientDescent, RMSProp" # default is Adam, for vanilla A3C is RMSProp
	# In information theory = the cross entropy between two probability distributions p and q over the same underlying set of events measures the average number of bits needed to identify an event drawn from the set.
	options["only_non_negative_entropy"] = True # "Cross-entropy and entropy are used for policy loss and if this flag is True, then entropy=max(0,entropy). If cross-entropy measures the average number of bits needed to identify an event, then it cannot be negative."
	# Use mean losses if max_batch_size is too big = in order to avoid NaN
	options["loss_type"] = "mean" # "type of loss reduction: sum, mean"
	options["policy_loss"] = "PPO" # "policy loss function: Vanilla, PPO, DISC"
	options["value_loss"] = "Vanilla" # "value loss function: Vanilla, PVO"
# Transition Prediction
	options["with_transition_predictor"] = False # Setting this option to True, you add an extra head to the default set of heads: actor, critic. This new head will be trained to predict r_t and the embedding of s_(t+1) given the embedding of s_t and a_t.
# PPO's and PVO's Loss clip range
	options["clip"] = 0.2 # "PPO/PVO initial clip range" # default is 0.2, for openAI is 0.1
	options["clip_decay"] = True # "Whether to decay the clip range"
	options["clip_annealing_function"] = "inverse_time_decay" # "annealing function: exponential_decay, inverse_time_decay, natural_exp_decay" # default is inverse_time_decay
	options["clip_decay_steps"] = 10**5 # "decay clip every x steps" # default is 10**6
	options["clip_decay_rate"] = 0.96 # "decay rate" # default is 0.25
# Importance Sampling Target
	options["importance_sampling_policy_target"] = 0.001 # "Importance Sampling target constant" -> Works only when policy_loss == DISC
# Learning rate
	options["alpha"] = 3.5e-4 # "initial learning rate" # default is 7.0e-4, for openAI is 2.5e-4
	options["alpha_decay"] = False # "whether to decay the learning rate"
	options["alpha_annealing_function"] = "exponential_decay" # "annealing function: exponential_decay, inverse_time_decay, natural_exp_decay" # default is inverse_time_decay
	options["alpha_decay_steps"] = 10**8 # "decay alpha every x steps" # default is 10**6
	options["alpha_decay_rate"] = 0.96 # "decay rate" # default is 0.25
# Intrinsic Rewards: Burda = Yuri = et al. "Exploration by Random Network Distillation." arXiv preprint arXiv:1810.12894 (2018).
	options["intrinsic_reward"] = False # "An intrinisc reward is given for exploring new states, and the agent is trained to maximize it."
	options["use_learnt_environment_model_as_observation"] = False # "Use the intrinsic reward weights (the learnt model of the environment) as network input."
	options["split_values"] = True # "Estimate separate values for extrinsic and intrinsic rewards." -> works also if intrinsic_reward=False
	options["intrinsic_reward_step"] = 2**20 # "Start using the intrinsic reward only when global step is greater than n."
	options["scale_intrinsic_reward"] = False # "Whether to scale the intrinsic reward with its standard deviation."
	options["intrinsic_rewards_mini_batch_fraction"] = 0 # "Keep only the best intrinsic reward in a mini-batch of size 'batch_size*fraction', and set other intrinsic rewards to 0."
	options["intrinsic_reward_gamma"] = 0.99 # "Discount factor for intrinsic rewards" # default is 0.95, for openAI is 0.99
	options["extrinsic_coefficient"] = 2. # "Scale factor for the extrinsic part of the advantage."
	options["intrinsic_coefficient"] = 1. # "Scale factor for the intrinsic part of the advantage."
	options["episodic_extrinsic_reward"] = True # "Bootstrap 0 for extrinsic value if state is terminal."
	options["episodic_intrinsic_reward"] = False # "Bootstrap 0 for intrinsic value if state is terminal."
# Experience Replay
	# Replay mean > 0 increases off-policyness
	options["replay_mean"] = 1 # "Mean number of experience replays per batch. Lambda parameter of a Poisson distribution. When replay_mean is 0, then experience replay is not active." # for A3C is 0, for ACER default is 4
	options["replay_step"] = 2**10 # "Start replaying experience when global step is greater than replay_step."
	options["replay_buffer_size"] = 2**9 # "Maximum number of batches stored in the experience buffer."
	options["replay_start"] = 1 # "Buffer minimum size before starting replay. Should be greater than 0 and lower than replay_buffer_size."
	options["replay_only_best_batches"] = False # "Whether to replay only those batches leading to a positive extrinsic reward (the best ones)."
	options["constraining_replay"] = False # "Use constraining replay loss for the Actor, in order to minimize the quadratic distance between the sampled batch actions and the Actor mean actions (softmax output)." -> might be useful only if combined with replay_only_best_batches=True
	options["train_critic_when_replaying"] = False # "Whether to train also the critic when replaying."
	options["recompute_value_when_replaying"] = False # "Whether to recompute value when replaying, using always up to date state values instead of old ones.", "Whether to recompute values, advantages and discounted cumulative rewards when replaying, even if not required by the model." # default True
	# options["loss_stationarity_range"] = 5e-3 # "Used to decide when to interrupt experience replay. If the mean actor loss is whithin this range, then no replay is performed."
# Prioritized Experience Replay: Schaul = Tom = et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).
	options["prioritization_scheme"] = "unclipped_gain_estimate" # The scheme to use for prioritized experience sampling. Use None to disable prioritized sampling. It works only when replay_mean > 0. One of the following: 'pruned_gain_estimate, clipped_gain_estimate, clipped_mean_gain_estimate, clipped_best_gain_estimate, unclipped_gain_estimate, unclipped_mean_gain_estimate, unclipped_best_gain_estimate, surprise, cumulative_extrinsic_return, transition_prediction_error'.
	options["prioritized_replay_alpha"] = 0.5 # "How much prioritization is used (0 - no prioritization = 1 - full prioritization)."
	options["prioritized_drop_probability"] = 1 # "Probability of removing the batch with the lowest priority instead of the oldest batch."
# Reward manipulators
	options["extrinsic_reward_manipulator"] = 'lambda x: x' # "Set to 'lambda x: x' for no manipulation. A lambda expression used to manipulate the extrinsic rewards."
	options["intrinsic_reward_manipulator"] = 'lambda x: x' # "Set to 'lambda x: x' for no manipulation. A lambda expression used to manipulate the intrinsic rewards."
# Actor-Critic parameters
	options["value_coefficient"] = 1 # "Value coefficient for tuning Critic learning rate." # default is 0.5
	options["environment_count"] = 32 # "Number of different parallel environments, used for training."
	options["groups_count"] = 4 # "Number n of groups. The environments are divided equally in n groups. Usually we have a thread per group. Used to better parallelize the training on the same machine."
	options["batch_size"] = 2**5 # "Maximum batch size." # default is 8
	# A big enough big_batch_size can significantly speed up the algorithm when training on GPU
	options["big_batch_size"] = 2**6 # "Number n > 0 of batches that compose a big-batch used for training. The bigger is n the more is the memory consumption."
	# Taking gamma < 1 introduces bias into the policy gradient estimate = regardless of the value function accuracy.
	options["gamma"] = 0.99 # "Discount factor for extrinsic rewards" # default is 0.95 = for openAI is 0.99
# Advantage Estimation
	options["advantage_estimator"] = "GAE_V" # "Can be one of the following: GAE, GAE_V, VTrace, Vanilla." # GAE_V and VTrace should reduce bias and variance when replay_ratio > 0
	# Taking lambda < 1 introduces bias only when the value function is inaccurate
	options["lambd"] = 0.95 # "It is the advantage estimator decay parameter used by GAE, GAE_V and VTrace." # Default for GAE is 0.95, default for VTrace is 1
# Entropy regularization
	options["entropy_regularization"] = True # "Whether to add entropy regularization to policy loss. Works only if intrinsic_reward == False" # default True
	options["beta"] = 1e-3 # "entropy regularization constant" # default is 0.001, for openAI is 0.01
# Log
	options["save_interval_step"] = 2**22 # "Save a checkpoint every n steps."
	# rebuild_network_after_checkpoint_is_saved may help saving RAM, but may be slow proportionally to save_interval_step.
	options["rebuild_network_after_checkpoint_is_saved"] = False # "Rebuild the whole network after checkpoint is saved. This may help saving RAM, but it's slow."
	options["max_checkpoint_to_keep"] = 3 # "Keep the last n checkpoints, delete the others"
	options["test_after_saving"] = False # "Whether to test after saving"
	options["print_test_results"] = False # "Whether to print test results when testing"
	options["episode_count_for_evaluation"] = 2**5 # "Number of matches used for evaluation scores"
	options["seconds_to_wait_for_printing_performance"] = 60 # "Number of seconds to wait for printing algorithm performance in terms of memory and time usage"
	options["checkpoint_dir"] = "./checkpoint" # "checkpoint directory"
	options["event_dir"] = "./events" # "events directory"
	options["log_directory"] = "./log" # "events directory"
	options["print_loss"] = True # "Whether to print losses inside statistics" # print_loss = True might slow down the algorithm
	options["print_policy_info"] = True # "Whether to print debug information about the actor inside statistics" # print_policy_info = True might slow down the algorithm
	options["show_episodes"] = 'random' # "What type of episodes to save: random, best, all, none"
	options["show_episode_probability"] = 5e-4 # "Probability of showing an episode when show_episodes == random"
	# save_episode_screen = True might slow down the algorithm -> use in combination with show_episodes = 'random' for best perfomance
	options["save_episode_screen"] = True # "Whether to save episode screens"
	# save_episode_gif = True slows down the algorithm, requires save_episode_screen, True to work
	options["save_episode_gif"] = True # "Whether to save episode GIF, requires save_episode_screen == True."
	options["gif_speed"] = 0.1 # "GIF frame speed in seconds."
	options["compress_gif"] = True # "Whether to zip the episode GIF."
	options["delete_screens_after_making_gif"] = True # "Whether to delete the screens after the GIF has been made."
	options["monitor_memory_usage"] = False # "Whether to monitor memory usage"
# Plot
	options["compute_plot_when_saving"] = True # "Whether to compute the plot when saving checkpoints"
	options["max_plot_size"] = 10 # "Maximum number of points in the plot. The smaller it is, the less RAM is required. If the log file has more than max_plot_size points, then max_plot_size means of slices are used instead."
	
def get():
	global options
	if not options:
		build()
	return SimpleNamespace(**options)
