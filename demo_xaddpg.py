# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
import multiprocessing
import json
import shutil
import ray

from xarl.agents.xaddpg import XADDPGTrainer, XADDPG_DEFAULT_CONFIG
from environments import *

# SELECT_ENV = "ToyExample-v0"
# SELECT_ENV = "CescoDrive-v2"
SELECT_ENV = "AlexDrive-v0"

CONFIG = XADDPG_DEFAULT_CONFIG.copy()
CONFIG.update({
	# "rollout_fragment_length": 1,
	# "train_batch_size": 2**7,
	# "learning_starts": 1500,
	# "grad_clip": None,
	"prioritized_replay": True,
	"num_envs_per_worker": 8, # Number of environments to evaluate vectorwise per worker. This enables model inference batching, which can improve performance for inference bottlenecked workloads.
	# "remote_worker_envs": True, # This will create env instances in Ray actors and step them in parallel. These remote processes introduce communication overheads, so this only helps if your env is very expensive to step / reset.
	# "num_workers": multiprocessing.cpu_count(),
	# "training_intensity": 2**4, # default is train_batch_size / rollout_fragment_length
	# "framework": "torch",
	########################################################
	"buffer_options": {
		'priority_id': "td_errors", # Which batch column to use for prioritisation. Default is inherited by DQN and it is 'td_errors'. One of the following: rewards, prev_rewards, td_errors.
		'priority_aggregation_fn': 'lambda x: np.mean(np.abs(x))', # A reduce function that takes as input a list of numbers and returns a number representing a batch priority.
		# 'cluster_size': None, # Default None, implying being equal to global_size. Maximum number of batches stored in a cluster (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'replay_sequence_length' (default is 1).
		'global_size': 2**13, # Default 50000. Maximum number of batches stored in all clusters (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'replay_sequence_length' (default is 1).
		'alpha': 0.6, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'beta': 0.4, # Parameter that regulates a mechanism for computing importance sampling.
		'epsilon': 1e-6, # Epsilon to add to a priority so that it is never equal to 0.
		'prioritized_drop_probability': 0, # Probability of dropping the batch having the lowest priority in the buffer.
		# 'update_insertion_time_when_sampling': False, # Default is False. Whether to update the insertion time batches to the time of sampling. It requires prioritized_drop_probability < 1. In DQN default is False.
		'global_distribution_matching': False, # If True then: At time t the probability of any experience being the max experience is 1/t regardless of when the sample was added, guaranteeing that at any given time the sampled experiences will approximately match the distribution of all samples seen so far.
		'prioritised_cluster_sampling_strategy': 'highest', # Whether to select which cluster to replay in a prioritised fashion. Four options: None; 'highest' - clusters with the highest priority are more likely to be sampled; 'average' - prioritise the cluster with priority closest to the average cluster priority; 'above_average' - prioritise the cluster with priority closest to the cluster with the smallest priority greater than the average cluster priority.
	},
	"batch_mode": "complete_episodes", # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes', otherwise it can also be 'truncate_episodes'.
	"clustering_scheme": "moving_best_extrinsic_reward_with_multiple_types", # Which scheme to use for building clusters. One of the following: none, extrinsic_reward, moving_best_extrinsic_reward, moving_best_extrinsic_reward_with_type, reward_with_type, reward_with_multiple_types, moving_best_extrinsic_reward_with_multiple_types.
	# "cluster_overview_size": None, # cluster_overview_size <= train_batch_size. If None, then it is automatically set to train_batch_size. -- When building a single train batch, do not sample a new cluster before x batches are sampled out of it. The closer is to train_batch_size, the faster is the algorithm.
	# "update_only_sampled_cluster": False, # Default is False. Whether to update the priority only in the sampled cluster and not in all, if the same batch is in more than one cluster. Setting this option to True causes a slighlty higher memory consumption but shall increase by far the speed in updating priorities.
})

####################################################################################
####################################################################################

ray.shutdown()
ray.init(ignore_reinit_error=True)

# Configure RLlib to train a policy using the “Taxi-v3” environment and a PPO optimizer
agent = XADDPGTrainer(CONFIG, env=SELECT_ENV)

# Inspect the trained policy and model, to see the results of training in detail
# policy = agent.get_policy()
# model = policy.model
# print(model.base_model.summary())

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

