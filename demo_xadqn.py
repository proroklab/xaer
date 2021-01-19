# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
import multiprocessing
import json
import shutil
import ray

from xarl.agents.xadqn import XADQNTrainer, XADQN_DEFAULT_CONFIG
from environments import *
from xarl.models.dqn import TFAdaptiveMultiHeadDQN
from ray.rllib.models import ModelCatalog
# Register the models to use.
ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadDQN)

# SELECT_ENV = "Taxi-v3"
# SELECT_ENV = "ToyExample-V0"
SELECT_ENV = "GridDrive-Hard"

CONFIG = XADQN_DEFAULT_CONFIG.copy()
CONFIG.update({
	# "log_level": "INFO",
	# "model": = {
	# 	"custom_model": "adaptive_multihead_network",
	# },
	"dueling": True,
	"double_q": True,
	# "n_step": 3,
	# "noisy": True,
	"num_atoms": 21,
	"v_max": 2**5,
	"v_min": -1,
	"rollout_fragment_length": 1,
	"train_batch_size": 2**7,
	"num_envs_per_worker": 8, # Number of environments to evaluate vectorwise per worker. This enables model inference batching, which can improve performance for inference bottlenecked workloads.
	"grad_clip": None,
	"learning_starts": 1500,
	# "hiddens": [512],
	# "exploration_config": {
	# 	"epsilon_timesteps": 2,
	# 	"final_epsilon": 0.0,
	# },
	# "remote_worker_envs": True, # This will create env instances in Ray actors and step them in parallel. These remote processes introduce communication overheads, so this only helps if your env is very expensive to step / reset.
	# "num_workers": multiprocessing.cpu_count(),
	# "training_intensity": 2**4, # default is train_batch_size / rollout_fragment_length
	# "framework": "torch",
	########################################################
	"batch_mode": "complete_episodes", # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes', otherwise it can also be 'truncate_episodes'.
	"prioritized_replay": True,
	########################################################
	"buffer_options": {
		'priority_id': "td_errors", # Which batch column to use for prioritisation. Default is inherited by DQN and it is 'td_errors'. One of the following: rewards, prev_rewards, td_errors.
		'priority_aggregation_fn': 'lambda x: np.mean(np.abs(x))', # A reduction that takes as input a list of numbers and returns a number representing a batch priority.
		'cluster_size': None, # Default None, implying being equal to global_size. Maximum number of batches stored in a cluster (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'replay_sequence_length' (default is 1).
		'global_size': 2**15, # Default 50000. Maximum number of batches stored in all clusters (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'replay_sequence_length' (default is 1).
		'min_cluster_size_proportion': 2, # Let X be the minimum cluster's size, and q be the min_cluster_size_proportion, then the cluster's size is guaranteed to be in [X, X+qX]. This shall help having a buffer reflecting the real distribution of tasks (where each task is associated to a cluster), thus avoiding over-estimation of task's priority.
		'alpha': 0.6, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'beta': 0.4, # To what degree to use importance weights (0 - no corrections, 1 - full correction).
		'eta': None, # A value > 0 that enables eta-weighting, thus allowing for importance weighting with priorities lower than 0. Eta is used to avoid importance weights equal to 0 when the sampled batch is the one with the highest priority. The closer eta is to 0, the closer to 0 would be the importance weight of the highest-priority batch.
		'epsilon': 1e-6, # Epsilon to add to a priority so that it is never equal to 0.
		'prioritized_drop_probability': 0, # Probability of dropping the batch having the lowest priority in the buffer instead of the one having the lowest timestamp. In DQN default is 0.
		'update_insertion_time_when_sampling': False, # Whether to update the insertion time batches to the time of sampling. It requires prioritized_drop_probability < 1. In DQN default is False.
		'global_distribution_matching': False, # Whether to use a random number rather than the batch priority during prioritised dropping. If True then: At time t the probability of any experience being the max experience is 1/t regardless of when the sample was added, guaranteeing that (when prioritized_drop_probability==1) at any given time the sampled experiences will approximately match the distribution of all samples seen so far.
		'prioritised_cluster_sampling_strategy': 'highest', # Whether to select which cluster to replay in a prioritised fashion -- 4 options: None; 'highest' - clusters with the highest priority are more likely to be sampled; 'average' - prioritise the cluster with priority closest to the average cluster priority; 'above_average' - prioritise the cluster with priority closest to the cluster with the smallest priority greater than the average cluster priority.
		'cluster_level_weighting': False, # Whether to use only cluster-level information to compute importance weights rather than the whole buffer.
	},
	"clustering_scheme": "multiple_types", # Which scheme to use for building clusters. One of the following: "none", "reward_against_zero", "reward_against_mean", "multiple_types_with_reward_against_mean", "type_with_reward_against_mean", "multiple_types", "type".
	"cluster_with_episode_type": False, # Useful with sparse-reward environments. Whether to cluster experience using information at episode-level.
	"cluster_overview_size": 1, # cluster_overview_size <= train_batch_size. If None, then cluster_overview_size is automatically set to train_batch_size. -- When building a single train batch, do not sample a new cluster before x batches are sampled from it. The closer cluster_overview_size is to train_batch_size, the faster is the batch sampling procedure.
})

####################################################################################
####################################################################################

ray.shutdown()
ray.init(ignore_reinit_error=True)

# Configure RLlib to train a policy using the “Taxi-v3” environment and a PPO optimizer
agent = XADQNTrainer(CONFIG, env=SELECT_ENV)

# Inspect the trained policy and model, to see the results of training in detail
policy = agent.get_policy()
model = policy.model
print(model.q_value_head.summary())
print(model.heads_model.summary())

# Train a policy. The following code runs 30 iterations and that’s generally enough to begin to see improvements in the “Taxi-v3” problem
# results = []
# episode_data = []
# episode_json = []
n = 0
while True:
	n += 1
	result = agent.train()
	# print(result)
	# results.append(result)
	episode = {
		'n': n, 
		'episode_reward_min': result['episode_reward_min'], 
		'episode_reward_mean': result['episode_reward_mean'], 
		'episode_reward_max': result['episode_reward_max'],  
		'episode_len_mean': result['episode_len_mean']
	}
	# episode_data.append(episode)
	# episode_json.append(json.dumps(episode))
	# file_name = agent.save(checkpoint_root)
	print(f'{n+1:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}, len mean: {result["episode_len_mean"]:8.4f}, train ratio: {(result["info"]["num_steps_trained"]/result["info"]["num_steps_sampled"]):8.4f}')
	# print(f'Checkpoint saved to {file_name}')

