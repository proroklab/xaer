# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
import multiprocessing
import json
import shutil
import ray

from xarl.agents.xappo import XAPPOTrainer, XAPPO_DEFAULT_CONFIG
from environments import *

# SELECT_ENV = "Taxi-v3"
# SELECT_ENV = "ToyExample-v0"
# SELECT_ENV = "CescoDrive-v2"
# SELECT_ENV = "AlexDrive-v0"
SELECT_ENV = "GridDrive-v1"

CONFIG = XAPPO_DEFAULT_CONFIG.copy()
CONFIG.update({
	"lambda": .95, # GAE(lambda) parameter. Taking lambda < 1 introduces bias only when the value function is inaccurate.
	# "clip_param": 0.2, # PPO surrogate loss options; default is 0.4. The higher it is, the higher the chances of catastrophic forgetting.
	########################################
	"batch_mode": "complete_episodes", # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes', otherwise it can also be 'truncate_episodes'.
	"vtrace": False, # Formula for computing the advantages: batch_mode==complete_episodes implies vtrace==False, thus gae==True.
	"replay_proportion": 2, # Set a p>0 to enable experience replay. Saved samples will be replayed with a p:1 proportion to new data samples.
	##########################################
	"gae_with_vtrace": False, # Formula for computing the advantages: combines GAE with V-Trace, for better sample efficiency.
	"prioritized_replay": True, # Whether to replay batches with the highest priority/importance/relevance for the agent.
	"update_priorities": True, # Whether to update priorities when replaying.
	"learning_starts": 100, # How many batches to sample before learning starts. Every batch has size 'rollout_fragment_length' (default is 50).
	"buffer_options": {
		'priority_id': "gains", # Which batch column to use for prioritisation. One of the following: gains, importance_weights, unweighted_advantages, advantages, rewards, prev_rewards, action_logp.
		'priority_aggregation_fn': 'np.sum', # A reduction that takes as input a list of numbers and returns a number representing a batch priority.
		'cluster_size': None, # Maximum number of batches stored in a cluster (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'rollout_fragment_length' (default is 50).
		'global_size': 2**9, # Maximum number of batches stored in all clusters (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'rollout_fragment_length' (default is 50).
		'alpha': 0.5, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'beta': None, # To what degree to use importance weights (0 - no corrections, 1 - full correction).
		'eta': 1e-2, # A value > 0 that enables eta-weighting, thus allowing for importance weighting with priorities lower than 0 if beta is > 0. Eta is used to avoid importance weights equal to 0 when the sampled batch is the one with the highest priority. The closer eta is to 0, the closer to 0 would be the importance weight of the highest-priority batch.
		'epsilon': 1e-6, # Epsilon to add to a priority so that it is never equal to 0.
		'prioritized_drop_probability': 0, # Probability of dropping the batch having the lowest priority in the buffer.
		'update_insertion_time_when_sampling': False, # Whether to update the insertion time batches to the time of sampling. It requires prioritized_drop_probability < 1. In DQN default is False.
		'global_distribution_matching': True, # Whether to use a random number rather than the batch priority during prioritised dropping. If True then: At time t the probability of any experience being the max experience is 1/t regardless of when the sample was added, guaranteeing that (when prioritized_drop_probability==1) at any given time the sampled experiences will approximately match the distribution of all samples seen so far.
		'prioritised_cluster_sampling_strategy': 'above_average', # Whether to select which cluster to replay in a prioritised fashion -- 4 options: None; 'highest' - clusters with the highest priority are more likely to be sampled; 'average' - prioritise the cluster with priority closest to the average cluster priority; 'above_average' - prioritise the cluster with priority closest to the cluster with the smallest priority greater than the average cluster priority.
		'cluster_level_weighting': False, # Whether to use only cluster-level information to compute importance weights rather than the whole buffer.
	},
	"clustering_scheme": "multiple_types_with_reward_against_mean", # Which scheme to use for building clusters. One of the following: "none", "reward_against_zero", "reward_against_mean", "multiple_types_with_reward_against_mean", "type_with_reward_against_mean", "multiple_types", "type".
	"cluster_with_episode_type": True, # Whether to cluster experience using information at episode-level.
	"cluster_overview_size": None, # cluster_overview_size <= train_batch_size. If None, then cluster_overview_size is automatically set to train_batch_size. -- When building a single train batch, do not sample a new cluster before x batches are sampled from it. The closer cluster_overview_size is to train_batch_size, the faster is the batch sampling procedure.
	"update_only_sampled_cluster": False, # If True, when sampling a batch from a cluster, no changes/updates to other clusters are performed if that batch is shared among these other clusters. Enabling this option would slightly speed-up batch sampling, by a constant proportional to the number of different clusters in the buffer.
})

####################################################################################
####################################################################################

from xarl.models.appo import TFAdaptiveMultiHeadNet
from ray.rllib.models import ModelCatalog
# Register the models to use.
ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadNet)
CONFIG["model"] = {
	"custom_model": "adaptive_multihead_network",
}

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
	print(f'{n+1:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}, len mean: {result["episode_len_mean"]:8.4f}, train ratio: {(result["info"]["num_steps_trained"]/result["info"]["num_steps_sampled"]):8.4f}')
	# print(f'Checkpoint saved to {file_name}')

