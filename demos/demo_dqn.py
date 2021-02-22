# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
import multiprocessing
import json
import shutil
import ray
import time
from xarl.utils.workflow import train

from ray.rllib.agents.dqn.dqn import DQNTrainer, DEFAULT_CONFIG as DQN_DEFAULT_CONFIG
from environments import *
from xarl.models.dqn import TFAdaptiveMultiHeadDQN
from ray.rllib.models import ModelCatalog
# Register the models to use.
ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadDQN)

# SELECT_ENV = "Taxi-v3"
# SELECT_ENV = "ToyExample-V0"
SELECT_ENV = "GridDrive-Hard"

CONFIG = DQN_DEFAULT_CONFIG.copy()
CONFIG.update({
	# "num_envs_per_worker": 2**3, # Number of environments to evaluate vectorwise per worker. This enables model inference batching, which can improve performance for inference bottlenecked workloads.
	"grad_clip": None,
	'buffer_size': 2**14, # Size of the experience buffer. Default 50000
	##################################
	"rollout_fragment_length": 2**6, # Divide episodes into fragments of this many steps each during rollouts.
	"replay_sequence_length": 1, # The number of contiguous environment steps to replay at once. This may be set to greater than 1 to support recurrent models.
	"train_batch_size": 2**8, # Number of transitions per train-batch
	"learning_starts": 1500, # How many batches to sample before learning starts. Every batch has size 'rollout_fragment_length' (default is 50).
	"prioritized_replay": True, # Whether to replay batches with the highest priority/importance/relevance for the agent.
	"batch_mode": "truncate_episodes", # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes', otherwise it can also be 'truncate_episodes'.
	"dueling": True,
	"double_q": True,
	"num_atoms": 21,
	"v_max": 2**5,
	"v_min": -1,
})

####################################################################################
####################################################################################

ray.shutdown()
ray.init(ignore_reinit_error=True)

train(DQNTrainer, CONFIG, SELECT_ENV, test_every_n_step=1000, stop_training_after_n_step=None)
