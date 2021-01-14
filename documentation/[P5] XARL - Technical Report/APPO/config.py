import multiprocessing
import sys
from pexpect import pxssh
import getpass
import json
import copy
import shlex
def copy_dict_and_update(d,u):
	new_dict = copy.deepcopy(d)
	new_dict.update(u)
	return new_dict

def copy_dict_and_update_with_key(d,k,u):
	new_dict = copy.deepcopy(d)
	new_dict[k].update(u)
	return new_dict

############################################################################################
############################################################################################

default_algorithm = 'XAPPO'
default_PPO_options = {
	# "model": {
	# 	"custom_model": "adaptive_multihead_network",
	# },
	# "lambda": .95, # GAE(lambda) parameter. Taking lambda < 1 introduces bias only when the value function is inaccurate.
	# "clip_param": 0.2, # PPO surrogate loss options; default is 0.4. The higher it is, the higher the chances of catastrophic forgetting.
	# "batch_mode": "complete_episodes", # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes', otherwise it can also be 'truncate_episodes'.
	# "vtrace": False, # Formula for computing the advantages: batch_mode==complete_episodes implies vtrace==False, thus gae==True.
	"rollout_fragment_length": 2**3, # Number of transitions per batch in the experience buffer
	"train_batch_size": 2**9, # Number of transitions per train-batch
	"replay_proportion": 4, # Set a p>0 to enable experience replay. Saved samples will be replayed with a p:1 proportion to new data samples.
	"replay_buffer_num_slots": 2**12, # Maximum number of batches stored in the experience buffer. Every batch has size 'rollout_fragment_length' (default is 50).
}
default_XAPPO_options = {
	# "model": {
	# 	"custom_model": "adaptive_multihead_network",
	# },
	# "lambda": .95, # GAE(lambda) parameter. Taking lambda < 1 introduces bias only when the value function is inaccurate.
	# "clip_param": 0.2, # PPO surrogate loss options; default is 0.4. The higher it is, the higher the chances of catastrophic forgetting.
	# "batch_mode": "complete_episodes", # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes', otherwise it can also be 'truncate_episodes'.
	# "vtrace": False, # Formula for computing the advantages: batch_mode==complete_episodes implies vtrace==False, thus gae==True.
	"rollout_fragment_length": 2**3, # Number of transitions per batch in the experience buffer
	"train_batch_size": 2**9, # Number of transitions per train-batch
	"replay_proportion": 4, # Set a p>0 to enable experience replay. Saved samples will be replayed with a p:1 proportion to new data samples.
	##########################################
	"gae_with_vtrace": False, # Formula for computing the advantages: it combines GAE with V-Trace, for better sample efficiency.
	"prioritized_replay": True, # Whether to replay batches with the highest priority/importance/relevance for the agent.
	"update_advantages_when_replaying": True, # Whether to recompute advantages when updating priorities.
	"learning_starts": 2**6, # How many batches to sample before learning starts. Every batch has size 'rollout_fragment_length' (default is 50).
	"buffer_options": {
		'priority_id': "gains", # Which batch column to use for prioritisation. One of the following: gains, advantages, rewards, prev_rewards, action_logp.
		'priority_aggregation_fn': 'np.mean', # A reduction function that takes as input a list of numbers and returns a number representing a batch priority.
		'cluster_size': None, # Maximum number of batches stored in a cluster (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'rollout_fragment_length' (default is 50).
		'global_size': 2**12, # Maximum number of batches stored in all clusters (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'rollout_fragment_length' (default is 50).
		'min_cluster_size_proportion': 1, # Let X be the minimum cluster's size, and q be the min_cluster_size_proportion, then the cluster's size is guaranteed to be in [X, X+qX]. This shall help having a buffer reflecting the real distribution of tasks (where each task is associated to a cluster), thus avoiding over-estimation of task's priority.
		'alpha': 0.5, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'beta': None, # To what degree to use importance weights (0 - no corrections, 1 - full correction).
		'eta': 1e-2, # A value > 0 that enables eta-weighting, thus allowing for importance weighting with priorities lower than 0 if beta is > 0. Eta is used to avoid importance weights equal to 0 when the sampled batch is the one with the highest priority. The closer eta is to 0, the closer to 0 would be the importance weight of the highest-priority batch.
		'epsilon': 1e-6, # Epsilon to add to a priority so that it is never equal to 0.
		'prioritized_drop_probability': 0, # Probability of dropping the batch having the lowest priority in the buffer.
		'update_insertion_time_when_sampling': False, # Whether to update the insertion time batches to the time of sampling. It requires prioritized_drop_probability < 1. In DQN default is False.
		'global_distribution_matching': False, # Whether to use a random number rather than the batch priority during prioritised dropping. If True then: At time t the probability of any experience being the max experience is 1/t regardless of when the sample was added, guaranteeing that (when prioritized_drop_probability==1) at any given time the sampled experiences will approximately match the distribution of all samples seen so far.
		'prioritised_cluster_sampling_strategy': 'highest', # Whether to select which cluster to replay in a prioritised fashion -- 4 options: None; 'highest' - clusters with the highest priority are more likely to be sampled; 'average' - prioritise the cluster with priority closest to the average cluster priority; 'above_average' - prioritise the cluster with priority closest to the cluster with the smallest priority greater than the average cluster priority.
		'cluster_level_weighting': False, # Whether to use only cluster-level information to compute importance weights rather than the whole buffer.
	},
	"clustering_scheme": "multiple_types", # Which scheme to use for building clusters. One of the following: "none", "reward_against_zero", "reward_against_mean", "multiple_types_with_reward_against_mean", "multiple_types_with_reward_against_zero", "type_with_reward_against_mean", "multiple_types", "type".
	"cluster_with_episode_type": False, # Whether to cluster experience using information at episode-level.
	"cluster_overview_size": 1, # cluster_overview_size <= train_batch_size. If None, then cluster_overview_size is automatically set to train_batch_size. -- When building a single train batch, do not sample a new cluster before x batches are sampled from it. The closer cluster_overview_size is to train_batch_size, the faster is the batch sampling procedure.
}

############################################################################################
############################################################################################
default_environment = 'GridDrive-v1'
grid_drive = [
	("gualtiero", ('APPO',default_environment,default_PPO_options)),
	("moschina", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'prioritized_replay':False}))),
	("margherita", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'clustering_scheme':'none', 'replay_proportion':0}))),
	("gretel", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'clustering_scheme':'none'}))),
	("dalibor", (default_algorithm,default_environment,default_XAPPO_options)),
	("donprocopio", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'clustering_scheme':'multiple_types_with_reward_against_mean'}))),
	("ernesto", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'clustering_scheme':'multiple_types_with_reward_against_zero'}))),
	("benes", (default_algorithm,default_environment,copy_dict_and_update_with_key(default_XAPPO_options, "buffer_options", {'prioritized_drop_probability':0.5}))),
	("fidelia", (default_algorithm,default_environment,copy_dict_and_update_with_key(default_XAPPO_options, "buffer_options", {'prioritised_cluster_sampling_strategy':None}))),
	("morales", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'replay_proportion':2}))),
	("dorina", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'replay_proportion':6}))),
	("pancrazio", (default_algorithm,default_environment,copy_dict_and_update_with_key(default_XAPPO_options, "buffer_options", {'min_cluster_size_proportion':0.5}))),
]
default_environment = 'GraphDrive-v0'
graph_drive = [
	("altoum", ('APPO',default_environment,default_PPO_options)),
	("ferrando", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'prioritized_replay':False}))),
	("dancairo", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'clustering_scheme':'none', 'replay_proportion':0}))),
	("roderigo", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'clustering_scheme':'none'}))),
	("eboli", (default_algorithm,default_environment,default_XAPPO_options)),
	("filindo", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'clustering_scheme':'multiple_types_with_reward_against_mean'}))),
	("elisabetta", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'clustering_scheme':'multiple_types_with_reward_against_zero'}))),
	("donpasquale", (default_algorithm,default_environment,copy_dict_and_update_with_key(default_XAPPO_options, "buffer_options", {'prioritized_drop_probability':0.5}))),
	("malatesta", (default_algorithm,default_environment,copy_dict_and_update_with_key(default_XAPPO_options, "buffer_options", {'prioritised_cluster_sampling_strategy':None}))),
	("doncurzio", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'replay_proportion':2}))),
	("bettina", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'replay_proportion':6}))),
	("ines", (default_algorithm,default_environment,copy_dict_and_update_with_key(default_XAPPO_options, "buffer_options", {'min_cluster_size_proportion':0.5}))),
]
default_environment = 'GridDrive-v1'
default_XAPPO_options = copy_dict_and_update_with_key(default_XAPPO_options, "buffer_options", {'beta':0.4})
grid_drive2 = [
	("edmondo", ('APPO',default_environment,default_PPO_options)),
	("zuniga", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'prioritized_replay':False}))),
	("mingo", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'clustering_scheme':'none', 'replay_proportion':0}))),
	("douphol", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'clustering_scheme':'none'}))),
	("eufemia", (default_algorithm,default_environment,default_XAPPO_options)),
	("lily", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'clustering_scheme':'multiple_types_with_reward_against_mean'}))),
	("edgar", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'clustering_scheme':'multiple_types_with_reward_against_zero'}))),
	("leonora", (default_algorithm,default_environment,copy_dict_and_update_with_key(default_XAPPO_options, "buffer_options", {'prioritized_drop_probability':0.5}))),
	("marullo", (default_algorithm,default_environment,copy_dict_and_update_with_key(default_XAPPO_options, "buffer_options", {'prioritised_cluster_sampling_strategy':None}))),
	("hansel", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'replay_proportion':2}))),
	("manrico", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'replay_proportion':6}))),
	("milada", (default_algorithm,default_environment,copy_dict_and_update_with_key(default_XAPPO_options, "buffer_options", {'min_cluster_size_proportion':0.5}))),
]
default_environment = 'GraphDrive-v0'
default_XAPPO_options = copy_dict_and_update_with_key(default_XAPPO_options, "buffer_options", {'beta':0.4})
graph_drive2 = [
	("brangania", ('APPO',default_environment,default_PPO_options)),
	("giovanna", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'prioritized_replay':False}))),
	("donbasilio", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'clustering_scheme':'none', 'replay_proportion':0}))),
	("grenvil", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'clustering_scheme':'none'}))),
	("donandronico", (default_algorithm,default_environment,default_XAPPO_options)),
	("remendado", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'clustering_scheme':'multiple_types_with_reward_against_mean'}))),
	("donbartolo", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'clustering_scheme':'multiple_types_with_reward_against_zero'}))),
	("frank", (default_algorithm,default_environment,copy_dict_and_update_with_key(default_XAPPO_options, "buffer_options", {'prioritized_drop_probability':0.5}))),
	("brander", (default_algorithm,default_environment,copy_dict_and_update_with_key(default_XAPPO_options, "buffer_options", {'prioritised_cluster_sampling_strategy':None}))),
	("lucia", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'replay_proportion':2}))),
	("fiorello", (default_algorithm,default_environment,copy_dict_and_update(default_XAPPO_options,{'replay_proportion':6}))),
	("berta", (default_algorithm,default_environment,copy_dict_and_update_with_key(default_XAPPO_options, "buffer_options", {'min_cluster_size_proportion':0.5}))),
]
experiment_list = grid_drive+graph_drive+grid_drive2+graph_drive2
