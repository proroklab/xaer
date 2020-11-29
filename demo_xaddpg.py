# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
import multiprocessing
import json
import shutil
import ray

# from ray.rllib.agents.ddpg.ddpg import DDPGTrainer, DEFAULT_CONFIG as DDPG_DEFAULT_CONFIG
from agents.xaddpg import XADDPGTrainer, XADDPG_DEFAULT_CONFIG
from environments import *

SELECT_ENV = "CescoDrive-v2"

CONFIG = XADDPG_DEFAULT_CONFIG.copy()
CONFIG["log_level"] = "WARN"
CONFIG["buffer_options"] = {
	'priority_id': "weights", # one of the following: gains, importance_weights, rewards, prev_rewards, action_logp
	'priority_aggregation_fn': 'lambda x: np.sum(np.abs(x))', # a reduce function (from a list of numbers to a number)
	'size': 50000, 
	'alpha': 0.6, 
	'beta': 0.4, # set to None for no weights correction
	'epsilon': 1e-4, # Epsilon to add to the TD errors when updating priorities.
	'prioritized_drop_probability': 1, 
	'global_distribution_matching': False, 
	'prioritised_cluster_sampling': False, 
}
CONFIG["clustering_scheme"] = "moving_best_extrinsic_reward_with_type" # one of the following: none, extrinsic_reward, moving_best_extrinsic_reward, moving_best_extrinsic_reward_with_type, reward_with_type
CONFIG["batch_mode"] = "complete_episodes" # can be equal to 'truncate_episodes' only when 'clustering_scheme' is 'none'

####################################################################################
####################################################################################

ray.shutdown()
ray.init(ignore_reinit_error=True)

# Configure a file location for checkpoints, in this case in a tmp/ppo/taxi subdirectory, deleting any previous files there
checkpoint_root = "tmp/ppo/taxi"
shutil.rmtree(checkpoint_root, ignore_errors=True, onerror=None)   # clean up old runs

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
	print(f'{n+1:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}, len mean: {result["episode_len_mean"]:8.4f}')
	# print(f'Checkpoint saved to {file_name}')

