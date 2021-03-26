# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
import multiprocessing
import json
import shutil
import ray
import time
from xarl.utils.workflow import train

from ray.rllib.agents.ppo.appo import APPOTrainer, DEFAULT_CONFIG as APPO_DEFAULT_CONFIG
from environments import *
from ray.rllib.models import ModelCatalog
from xarl.models.appo import TFAdaptiveMultiHeadNet
ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadNet)

# SELECT_ENV = "Taxi-v3"
# SELECT_ENV = "ToyExample-V0"
# SELECT_ENV = "CescoDrive-V1"
# SELECT_ENV = "GridDrive-Hard"
SELECT_ENV = "GraphDrive-Hard"

CONFIG = APPO_DEFAULT_CONFIG.copy()
CONFIG.update({
	"seed": 42, # This makes experiments reproducible.
	"rollout_fragment_length": 2**3, # Number of transitions per batch in the experience buffer
	"train_batch_size": 2**9, # Number of transitions per train-batch
	"replay_proportion": 4, # Set a p>0 to enable experience replay. Saved samples will be replayed with a p:1 proportion to new data samples.
	"replay_buffer_num_slots": 2**12, # Maximum number of batches stored in the experience buffer. Every batch has size 'rollout_fragment_length' (default is 50).
})

####################################################################################
####################################################################################

ray.shutdown()
ray.init(ignore_reinit_error=True, address='auto')

train(APPOTrainer, CONFIG, SELECT_ENV, test_every_n_step=1e6, stop_training_after_n_step=None)
