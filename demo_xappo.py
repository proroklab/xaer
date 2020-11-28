# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
import multiprocessing
import json
import shutil
import ray

from agents.xappo import XAPPOTrainer, XAPPO_DEFAULT_CONFIG
from environments import *

# SELECT_ENV = "ToyExample-v0"
SELECT_ENV = "CescoDrive-v0"
N_ITER = 30
CONFIG = XAPPO_DEFAULT_CONFIG.copy()
CONFIG["log_level"] = "WARN"
CONFIG["replay_proportion"] = 1 # The input batch will be returned and an additional number of batches proportional to this value will be added as well.
CONFIG["lambda"] = .95 # GAE(lambda) parameter
CONFIG["clip_param"] = 0.2 # PPO surrogate loss options
CONFIG["clustering_scheme"] = "moving_best_extrinsic_reward_with_type" # one of the following: none, extrinsic_reward, moving_best_extrinsic_reward, moving_best_extrinsic_reward_with_type, reward_with_type
CONFIG["gae_with_vtrace"] = True # combines GAE with V-Tracing
###############################################
# Priority_weight: For XAPPO one of the following: gains, importance_weights, advantages, rewards, prev_rewards, action_logp
CONFIG["priority_weight"] = "gains"
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
agent = XAPPOTrainer(CONFIG, env=SELECT_ENV)

# Inspect the trained policy and model, to see the results of training in detail
policy = agent.get_policy()
model = policy.model
print(model.base_model.summary())

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

