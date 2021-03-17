# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
import multiprocessing
import json
import shutil
import ray
import time
from xarl.utils.workflow import train

from xarl.agents.xappo import XAPPOTrainer, XAPPO_DEFAULT_CONFIG
from environments import *
from xarl.models.appo import TFAdaptiveMultiHeadNet
from ray.rllib.models import ModelCatalog
# Register the models to use.
ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadNet)

# SELECT_ENV = "Taxi-v3"
# SELECT_ENV = "ToyExample-V0"
# SELECT_ENV = "CescoDrive-V1"
SELECT_ENV = "GraphDrive-Hard"
# SELECT_ENV = "GridDrive-Hard"

CONFIG = XAPPO_DEFAULT_CONFIG.copy()
CONFIG.update({
	"rollout_fragment_length": 2**3, # Number of transitions per batch in the experience buffer
	"train_batch_size": 2**9, # Number of transitions per train-batch
	"replay_proportion": 4, # Set a p>0 to enable experience replay. Saved samples will be replayed with a p:1 proportion to new data samples.
	"replay_buffer_num_slots": 2**12, # Maximum number of batches stored in the experience buffer. Every batch has size 'rollout_fragment_length' (default is 50).	
	###################################
	"gae_with_vtrace": False, # Useful when default "vtrace" is not active. Formula for computing the advantages: it combines GAE with V-Trace.
	"prioritized_replay": True, # Whether to replay batches with the highest priority/importance/relevance for the agent.
	"update_advantages_when_replaying": True, # Whether to recompute advantages when updating priorities.
	"learning_starts": 2**6, # How many batches to sample before learning starts. Every batch has size 'rollout_fragment_length' (default is 50).
	"buffer_options": {
		'priority_id': 'gains', # Which batch column to use for prioritisation. One of the following: gains, advantages, rewards, prev_rewards, action_logp.
		'priority_lower_limit': None, # A value lower than the lowest possible priority. It depends on the priority_id. By default in DQN and DDPG it is td_error 0, while in PPO it is gain None.
		'priority_aggregation_fn': 'np.mean', # A reduction that takes as input a list of numbers and returns a number representing a batch priority.
		'cluster_size': None, # Maximum number of batches stored in a cluster (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'rollout_fragment_length' (default is 50).
		'global_size': 2**12, # Maximum number of batches stored in all clusters (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'rollout_fragment_length' (default is 50).
		'min_cluster_size_proportion': 1, # Let X be the minimum cluster's size, and q be the min_cluster_size_proportion, then the cluster's size is guaranteed to be in [X, X+qX]. This shall help having a buffer reflecting the real distribution of tasks (where each task is associated to a cluster), thus avoiding over-estimation of task's priority.
		'prioritization_alpha': 0.6, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'prioritization_importance_beta': 0.4, # To what degree to use importance weights (0 - no corrections, 1 - full correction).
		'prioritization_importance_eta': 1e-2, # Used only if priority_lower_limit is None. A value > 0 that enables eta-weighting, thus allowing for importance weighting with priorities lower than 0 if beta is > 0. Eta is used to avoid importance weights equal to 0 when the sampled batch is the one with the highest priority. The closer eta is to 0, the closer to 0 would be the importance weight of the highest-priority batch.
		'prioritization_epsilon': 1e-6, # prioritization_epsilon to add to a priority so that it is never equal to 0.
		'prioritized_drop_probability': 0, # Probability of dropping the batch having the lowest priority in the buffer.
		'global_distribution_matching': False, # Whether to use a random number rather than the batch priority during prioritised dropping. If True then: At time t the probability of any experience being the max experience is 1/t regardless of when the sample was added, guaranteeing that (when prioritized_drop_probability==1) at any given time the sampled experiences will approximately match the distribution of all samples seen so far.
		'cluster_prioritisation_strategy': 'highest', # Whether to select which cluster to replay in a prioritised fashion -- 4 options: None; 'highest' - clusters with the highest priority are more likely to be sampled; 'average' - prioritise the cluster with priority closest to the average cluster priority; 'above_average' - prioritise the cluster with priority closest to the cluster with the smallest priority greater than the average cluster priority.
		'cluster_level_weighting': True, # Whether to use only cluster-level information to compute importance weights rather than the whole buffer.
	},
	"clustering_scheme": "multiple_types_with_reward_against_mean", # Which scheme to use for building clusters. One of the following: "none", "reward_against_zero", "reward_against_mean", "multiple_types_with_reward_against_mean", "multiple_types_with_reward_against_zero", "type_with_reward_against_mean", "multiple_types", "type".
	"cluster_with_episode_type": False, # Most useful with sparse-reward environments. Whether to cluster experience using information at episode-level. It requires "batch_mode" == "complete_episodes".
	"cluster_overview_size": 1, # cluster_overview_size <= train_batch_size. If None, then cluster_overview_size is automatically set to train_batch_size. -- When building a single train batch, do not sample a new cluster before x batches are sampled from it. The closer cluster_overview_size is to train_batch_size, the faster is the batch sampling procedure.
	"collect_cluster_metrics": False, # Whether to collect metrics about the experience clusters. It consumes more resources.
	"sample_also_from_buffer_of_recent_elements": False, # Whether to sample in a randomised fashion from both a non-prioritised buffer of most recent elements and the XA prioritised buffer.
})

####################################################################################
####################################################################################

ray.shutdown()
ray.init(ignore_reinit_error=True)

train(XAPPOTrainer, CONFIG, SELECT_ENV, test_every_n_step=1e6, stop_training_after_n_step=None)
