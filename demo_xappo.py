# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
import multiprocessing
import json
import shutil
import ray

from xarl.agents.xappo import XAPPOTrainer, XAPPO_DEFAULT_CONFIG, GAINS
from environments import *

# SELECT_ENV = "Taxi-v3"
# SELECT_ENV = "ToyExample-v0"
# SELECT_ENV = "CescoDrive-v2"
# SELECT_ENV = "AlexDrive-v0"
SELECT_ENV = "GridDrive-v1"

CONFIG = XAPPO_DEFAULT_CONFIG.copy()
CONFIG["log_level"] = "WARN"
CONFIG["lambda"] = .95 # GAE(lambda) parameter
CONFIG["clip_param"] = 0.2 # PPO surrogate loss options
# CONFIG["gamma"] = 0.999
##################################
# For more config options, see here: https://docs.ray.io/en/master/rllib-algorithms.html#asynchronous-proximal-policy-optimization-appo
CONFIG["replay_proportion"] = 1 # Set a p>0 to enable experience replay. Saved samples will be replayed with a p:1 proportion to new data samples.
CONFIG["learning_starts"] = 1000 # How many batches to sample before learning starts.
CONFIG["prioritized_replay"] = True
CONFIG["buffer_options"] = {
	'priority_id': GAINS, # Which batch column to use for prioritisation. One of the following: gains, importance_weights, advantages, rewards, prev_rewards, action_logp
	'priority_aggregation_fn': 'np.sum', # A reduce function that takes as input a list of numbers and returns a number representing a batch's priority
	'size': 2**9, # "Maximum number of batches stored in the experience buffer."
	'alpha': 0.5, # "How much prioritization is used (0 - no prioritization, 1 - full prioritization)."
	'beta': None, # Parameter that regulates a mechanism for computing importance sampling. Not needed in PPO.
	'epsilon': 1e-6, # Epsilon to add to the TD errors when updating priorities.
	'prioritized_drop_probability': 0.5, # Probability of dropping experience with the lowest priority in the buffer
	'global_distribution_matching': False, # "If True, then: At time t the probability of any experience being the max experience is 1/t regardless of when the sample was added, guaranteeing that at any given time the sampled experiences will approximately match the distribution of all samples seen so far."
	'prioritised_cluster_sampling': False, # Whether to select which cluster to replay in a prioritised fashion
}
CONFIG["clustering_scheme"] = "moving_best_extrinsic_reward_with_multiple_types" # Which scheme to use for building clusters. One of the following: none, extrinsic_reward, moving_best_extrinsic_reward, moving_best_extrinsic_reward_with_type, reward_with_type, reward_with_multiple_types, moving_best_extrinsic_reward_with_multiple_types
CONFIG["batch_mode"] = "complete_episodes" # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes' otherwise it can also be 'truncate_episodes'
CONFIG["vtrace"] = False # Formula for computing the advantage: batch_mode==complete_episodes implies vtrace==False
CONFIG["gae_with_vtrace"] = True # Formula for computing the advantage: combines GAE with V-Tracing

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

