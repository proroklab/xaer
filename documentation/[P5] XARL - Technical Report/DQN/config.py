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

default_environment = 'GridDrive-v1'
default_dqn_options = {
	"model": {
		"custom_model": "adaptive_multihead_network",
	},
	"dueling": True,
	"double_q": True,
	# "n_step": 3,
	# "noisy": True,
	"prioritized_replay": True,
	"num_atoms": 21,
	"v_max": 2**5,
	"v_min": -1,
	"rollout_fragment_length": 1,
	"train_batch_size": 2**7,
	"num_envs_per_worker": 8, # Number of environments to evaluate vectorwise per worker. This enables model inference batching, which can improve performance for inference bottlenecked workloads.
	"grad_clip": None,
	"learning_starts": 1500,
	#############################
	"buffer_size": 2**15, # Size of the experience buffer. Default 50000
	"batch_mode": "complete_episodes", # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes', otherwise it can also be 'truncate_episodes'.
}
default_xadqn_options_p1 = {
	"model": {
		"custom_model": "adaptive_multihead_network",
	},
	"dueling": True,
	"double_q": True,
	# "n_step": 3,
	# "noisy": True,
	"prioritized_replay": True,
	"num_atoms": 21,
	"v_max": 2**5,
	"v_min": -1,
	"rollout_fragment_length": 1,
	"train_batch_size": 2**7,
	"num_envs_per_worker": 8, # Number of environments to evaluate vectorwise per worker. This enables model inference batching, which can improve performance for inference bottlenecked workloads.
	"grad_clip": None,
	"learning_starts": 1500,
	#############################
	"buffer_options": {
		'priority_id': "td_errors", # Which batch column to use for prioritisation. Default is inherited by DQN and it is 'td_errors'. One of the following: rewards, prev_rewards, td_errors.
		'priority_aggregation_fn': 'lambda x: np.mean(np.abs(x))', # A reduce function that takes as input a list of numbers and returns a number representing a batch priority.
		'cluster_size': None, # Default None, implying being equal to global_size. Maximum number of batches stored in a cluster (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'replay_sequence_length' (default is 1).
		'global_size': 2**15, # Default 50000. Maximum number of batches stored in all clusters (which number depends on the clustering scheme) of the experience buffer. Every batch has size 'replay_sequence_length' (default is 1).
		'alpha': 0.6, # How much prioritization is used (0 - no prioritization, 1 - full prioritization).
		'beta': 0.4, # Parameter that regulates a mechanism for computing importance sampling.
		'epsilon': 1e-6, # Epsilon to add to a priority so that it is never equal to 0.
		'prioritized_drop_probability': 0, # Probability of dropping the batch having the lowest priority in the buffer.
		'update_insertion_time_when_sampling': False, # Default is False. Whether to update the insertion time batches to the time of sampling. It requires prioritized_drop_probability < 1. In DQN default is False.
		'global_distribution_matching': False, # If True then: At time t the probability of any experience being the max experience is 1/t regardless of when the sample was added, guaranteeing that at any given time the sampled experiences will approximately match the distribution of all samples seen so far.
		'prioritised_cluster_sampling_strategy': 'highest', # Whether to select which cluster to replay in a prioritised fashion. Four options: None; 'highest' - clusters with the highest priority are more likely to be sampled; 'average' - prioritise the cluster with priority closest to the average cluster priority; 'above_average' - prioritise the cluster with priority closest to the cluster with the smallest priority greater than the average cluster priority.
		'cluster_level_weighting': True, # Whether to use only cluster-level information to compute importance weights rather than information about the whole buffer.
	},
	"batch_mode": "complete_episodes", # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes', otherwise it can also be 'truncate_episodes'.
	"clustering_scheme": "multiple_types", # Which scheme to use for building clusters. One of the following: "none", "reward_against_zero", "reward_against_mean", "multiple_types_with_reward_against_mean", "type_with_reward_against_mean", "multiple_types", "type".
	"cluster_with_episode_type": False, # Whether to cluster experience using information at episode-level.
	"cluster_overview_size": 1, # cluster_overview_size <= train_batch_size -- When building a single train batch, do not sample a new cluster before x batches are sampled out of it. The closer is to train_batch_size, the faster is the algorithm. If None, then it is automatically set to train_batch_size.
	"update_only_sampled_cluster": False, # Default is False. Whether to update the priority only in the sampled cluster and not in all, if the same batch is in more than one cluster. Setting this option to True causes a slighlty higher memory consumption but shall increase by far the speed in updating priorities.
}
# P2
default_xadqn_options_p2 = copy_dict_and_update_with_key(default_xadqn_options_p1, "buffer_options", {
	'cluster_level_weighting': False,
})
# P3
default_xadqn_options_p3 = copy_dict_and_update(default_xadqn_options_p1, {
	'cluster_overview_size': None,
})
# P4
default_xadqn_options_p4 = copy_dict_and_update(default_xadqn_options_p1, {
	'update_only_sampled_cluster': True,
})

############################################################################################
############################################################################################

baseline = [
	("gualtiero", ('DQN',default_environment,default_dqn_options))
]
p1 = [
	("dalibor", ('XADQN',default_environment,default_xadqn_options_p1)),
	("gretel", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p1,{'clustering_scheme':'none'}))),
	("moschina", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p1,{'clustering_scheme':'reward_against_mean'}))),
	("benes", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p1,{'clustering_scheme':'reward_against_zero'}))),
	("donprocopio", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p1,{'clustering_scheme':'multiple_types_with_reward_against_mean'}))),
	("remendado", ('XADQN',default_environment,copy_dict_and_update_with_key(default_xadqn_options_p1,'buffer_options',{'global_distribution_matching':True}))),
	("donandronico", ('XADQN',default_environment,copy_dict_and_update_with_key(default_xadqn_options_p1,'buffer_options',{'prioritized_drop_probability':0.5}))),
	("grenvil", ('XADQN',default_environment,copy_dict_and_update_with_key(default_xadqn_options_p1,'buffer_options',{'prioritized_drop_probability':1}))),
	# ("remendado", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p1,{'cluster_overview_size':2**0}))),
	# ("donandronico", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p1,{'cluster_overview_size':2**4}))),
	# ("grenvil", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p1,{'cluster_overview_size':None}))),
]
p2 = [
	("dorina", ('XADQN',default_environment,default_xadqn_options_p2)),
	("morales", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p2,{'clustering_scheme':'none'}))),
	("edmondo", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p2,{'clustering_scheme':'reward_against_mean'}))),
	("zuniga", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p2,{'clustering_scheme':'reward_against_zero'}))),
	("mingo", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p2,{'clustering_scheme':'multiple_types_with_reward_against_mean'}))),
	("donbasilio", ('XADQN',default_environment,copy_dict_and_update_with_key(default_xadqn_options_p2,'buffer_options',{'global_distribution_matching':True}))),
	("giovanna", ('XADQN',default_environment,copy_dict_and_update_with_key(default_xadqn_options_p2,'buffer_options',{'prioritized_drop_probability':0.5}))),
	("pancrazio", ('XADQN',default_environment,copy_dict_and_update_with_key(default_xadqn_options_p2,'buffer_options',{'prioritized_drop_probability':1}))),
	# ("donbasilio", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p2,{'cluster_overview_size':2**0}))),
	# ("giovanna", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p2,{'cluster_overview_size':2**4}))),
	# ("pancrazio", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p2,{'cluster_overview_size':None}))),
]
p3 = [
	("eufemia", ('XADQN',default_environment,default_xadqn_options_p3)),
	("lily", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p3,{'clustering_scheme':'none'}))),
	("leonora", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p3,{'clustering_scheme':'reward_against_mean'}))),
	("marullo", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p3,{'clustering_scheme':'reward_against_zero'}))),
	("hansel", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p3,{'clustering_scheme':'multiple_types_with_reward_against_mean'}))),
	("berta", ('XADQN',default_environment,copy_dict_and_update_with_key(default_xadqn_options_p3,'buffer_options',{'global_distribution_matching':True}))),
	("lucia", ('XADQN',default_environment,copy_dict_and_update_with_key(default_xadqn_options_p3,'buffer_options',{'prioritized_drop_probability':0.5}))),
	("fiorello", ('XADQN',default_environment,copy_dict_and_update_with_key(default_xadqn_options_p3,'buffer_options',{'prioritized_drop_probability':1}))),
	# ("berta", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p3,{'cluster_overview_size':2**0}))),
	# ("lucia", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p3,{'cluster_overview_size':2**4}))),
	# ("fiorello", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p3,{'cluster_overview_size':None}))),
]
p4 = [
	("bettina", ('XADQN',default_environment,default_xadqn_options_p4)),
	("doncurzio", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p4,{'clustering_scheme':'none'}))),
	("donpasquale", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p4,{'clustering_scheme':'reward_against_mean'}))),
	("malatesta", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p4,{'clustering_scheme':'reward_against_zero'}))),
	("filindo", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p4,{'clustering_scheme':'multiple_types_with_reward_against_mean'}))),
	("brander", ('XADQN',default_environment,copy_dict_and_update_with_key(default_xadqn_options_p4,'buffer_options',{'global_distribution_matching':True}))),
	("eboli", ('XADQN',default_environment,copy_dict_and_update_with_key(default_xadqn_options_p4,'buffer_options',{'prioritized_drop_probability':0.5}))),
	("ines", ('XADQN',default_environment,copy_dict_and_update_with_key(default_xadqn_options_p4,'buffer_options',{'prioritized_drop_probability':1}))),
	# ("brander", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p4,{'cluster_overview_size':2**0}))),
	# ("eboli", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p4,{'cluster_overview_size':2**4}))),
	# ("ines", ('XADQN',default_environment,copy_dict_and_update(default_xadqn_options_p4,{'cluster_overview_size':None}))),
]
experiment_list = baseline+p1+p2+p3+p4
