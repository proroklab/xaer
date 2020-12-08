# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
import multiprocessing
import json
import shutil
import ray

from ray.rllib.agents.dqn.dqn import DQNTrainer, DEFAULT_CONFIG as DQN_DEFAULT_CONFIG
from xarl.agents.xadqn import DQNTrainer as DQNTrainer_ExplainedVar
from environments import *

# SELECT_ENV = "ToyExample-v0"
SELECT_ENV = "AlexDrive-v2" # discrete

CONFIG = DQN_DEFAULT_CONFIG.copy()
CONFIG["log_level"] = "WARN"
CONFIG["batch_mode"] = "complete_episodes" # Whether to rollout "complete_episodes" or "truncate_episodes" to `rollout_fragment_length` length unrolls. Episode truncation guarantees evenly sized batches, but increases variance as the reward-to-go will need to be estimated at truncation boundaries.

####################################################################################
####################################################################################

ray.shutdown()
ray.init(ignore_reinit_error=True)

# Configure RLlib to train a policy using the “Taxi-v3” environment and a PPO optimizer
# agent = DDPGTrainer(CONFIG, env=SELECT_ENV)
agent = DQNTrainer_ExplainedVar(CONFIG, env=SELECT_ENV)

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

