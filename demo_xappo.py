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
SELECT_ENV = "AlexDrive-v0"
# SELECT_ENV = "GridDrive-v1"

CONFIG = XAPPO_DEFAULT_CONFIG.copy()
CONFIG["log_level"] = "WARN"
CONFIG["lambda"] = .95 # GAE(lambda) parameter. Taking lambda < 1 introduces bias only when the value function is inaccurate.
# CONFIG["gamma"] = 0.99 # Default is 0.99 - 1: future rewards are more important; 0+epsilon: immediate rewards are more important.
CONFIG["clip_param"] = 0.2 # PPO surrogate loss options; default is 0.4. The higher it is, the higher the chances of catastrophic forgetting.
# CONFIG["rollout_fragment_length"] = 50 # The maximum (it is not also the minimum only when 'batch_mode' == 'complete_episodes') size of a single batch, in terms of state transitions. Default is 50.
# CONFIG["train_batch_size"] = 500 # The size of a batch of batches used for training, in terms of state transitions. Default is 500.
##################################
# For more config options, see here: https://docs.ray.io/en/master/rllib-algorithms.html#asynchronous-proximal-policy-optimization-appo
CONFIG["replay_proportion"] = 1 # Set a p>0 to enable experience replay. Saved samples will be replayed with a p:1 proportion to new data samples.
CONFIG["learning_starts"] = 100 # How many batches to sample before learning starts.
CONFIG["prioritized_replay"] = True
CONFIG["buffer_options"] = {
	'priority_id': GAINS, # Which batch column to use for prioritisation. One of the following: gains, importance_weights, unweighted_advantages, advantages, rewards, prev_rewards, action_logp.
	'priority_aggregation_fn': 'np.sum', # A reduce function that takes as input a list of numbers and returns a number representing a batch priority.
	'size': 2**8, # Maximum number of batches stored in a cluster (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'rollout_fragment_length' (default is 50).
	'alpha': 0.5, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
	'beta': None, # Parameter that regulates a mechanism for computing importance sampling; PPO probably does not need it.
	'epsilon': 1e-6, # Epsilon to add to a priority so that it is never equal to 0.
	'prioritized_drop_probability': 0, # Probability of dropping the batch having the lowest priority in the buffer.
	'global_distribution_matching': False, # If True then: At time t the probability of any experience being the max experience is 1/t regardless of when the sample was added, guaranteeing that at any given time the sampled experiences will approximately match the distribution of all samples seen so far.
	'prioritised_cluster_sampling': True, # Whether to select which cluster to replay in a prioritised fashion.
}
CONFIG["clustering_scheme"] = "moving_best_extrinsic_reward_with_multiple_types" # Which scheme to use for building clusters. One of the following: none, extrinsic_reward, moving_best_extrinsic_reward, moving_best_extrinsic_reward_with_type, reward_with_type, reward_with_multiple_types, moving_best_extrinsic_reward_with_multiple_types.
CONFIG["batch_mode"] = "complete_episodes" # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes', otherwise it can also be 'truncate_episodes'.
CONFIG["vtrace"] = False # Formula for computing the advantages: batch_mode==complete_episodes implies vtrace==False.
CONFIG["gae_with_vtrace"] = False # Formula for computing the advantages: combines GAE with V-Trace, for better sample efficiency.

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

