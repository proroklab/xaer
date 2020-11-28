# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
import multiprocessing
import json
import shutil
import ray

from ray.rllib.agents.ddpg.ddpg import DDPGTrainer, DEFAULT_CONFIG as DDPG_DEFAULT_CONFIG
# from agents.xaddpg import XADDPGTrainer, XADDPG_DEFAULT_CONFIG
from environments import *

SELECT_ENV = "CescoDrive-v2"
N_ITER = 30
CONFIG = DDPG_DEFAULT_CONFIG.copy()
CONFIG["log_level"] = "WARN"
CONFIG["clustering_scheme"] = "moving_best_extrinsic_reward_with_type" # one of the following: none, extrinsic_reward, moving_best_extrinsic_reward, moving_best_extrinsic_reward_with_type, reward_with_type
###############################################
# Priority_weight: For XADQN one of the following: weigths, rewards, prev_rewards, action_logp
CONFIG["priority_weight"] = "weights"
###############################################
CONFIG["priority_weights_aggregator"] = 'np.mean' # a reduce function (from a list of numbers to a number)

####################################################################################
####################################################################################

ray.shutdown()
ray.init(ignore_reinit_error=True)

# Configure a file location for checkpoints, in this case in a tmp/ppo/taxi subdirectory, deleting any previous files there
checkpoint_root = "tmp/ppo/taxi"
shutil.rmtree(checkpoint_root, ignore_errors=True, onerror=None)   # clean up old runs

# Configure RLlib to train a policy using the “Taxi-v3” environment and a PPO optimizer
agent = DDPGTrainer(CONFIG, env=SELECT_ENV)

# Inspect the trained policy and model, to see the results of training in detail
# policy = agent.get_policy()
# model = policy.model
# print(model.base_model.summary())

# Train a policy. The following code runs 30 iterations and that’s generally enough to begin to see improvements in the “Taxi-v3” problem
# results = []
# episode_data = []
# episode_json = []
for n in range(N_ITER):
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

