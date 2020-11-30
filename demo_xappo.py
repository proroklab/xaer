# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
import multiprocessing
import json
import shutil
import ray

from agents.xappo import XAPPOTrainer, XAPPO_DEFAULT_CONFIG, GAINS
from environments import *

# SELECT_ENV = "ToyExample-v0"
# SELECT_ENV = "CescoDrive-v2"
SELECT_ENV = "AlexDrive-v0"

CONFIG = XAPPO_DEFAULT_CONFIG.copy()
CONFIG["log_level"] = "WARN"
CONFIG["lambda"] = .95 # GAE(lambda) parameter
CONFIG["clip_param"] = 0.2 # PPO surrogate loss options
# CONFIG["gamma"] = 0.999
##################################
CONFIG["replay_proportion"] = 1
CONFIG["learning_starts"] = 1000 # How many steps of the model to sample before learning starts.
CONFIG["prioritized_replay"] = True
CONFIG["replay_sequence_length"] = 1
CONFIG["buffer_options"] = {
	'priority_id': GAINS, # one of the following: gains, importance_weights, rewards, prev_rewards, action_logp
	'priority_aggregation_fn': 'np.sum', # a reduce function (from a list of numbers to a number)
	'size': 2**9, 
	'alpha': 0.5, 
	'beta': None, 
	'epsilon': 1e-6, # Epsilon to add to the TD errors when updating priorities.
	'prioritized_drop_probability': 1, 
	'global_distribution_matching': False, 
	'prioritised_cluster_sampling': False, 
}
# Clustering Scheme
CONFIG["clustering_scheme"] = "moving_best_extrinsic_reward_with_type" # one of the following: none, extrinsic_reward, moving_best_extrinsic_reward, moving_best_extrinsic_reward_with_type, reward_with_type
CONFIG["batch_mode"] = "complete_episodes" # can be equal to 'truncate_episodes' only when 'clustering_scheme' is 'none'
CONFIG["vtrace"] = False # batch_mode==complete_episodes implies vtrace==False
CONFIG["gae_with_vtrace"] = True # combines GAE with V-Tracing

####################################################################################
####################################################################################

ray.shutdown()
ray.init(ignore_reinit_error=True)

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

