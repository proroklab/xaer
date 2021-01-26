# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
import multiprocessing
import json
import shutil
import ray
import time

from ray.rllib.agents.ppo.appo import APPOTrainer, DEFAULT_CONFIG as APPO_DEFAULT_CONFIG
from environments import *
from ray.rllib.models import ModelCatalog
from xarl.models.appo import TFAdaptiveMultiHeadNet
ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadNet)

# SELECT_ENV = "Taxi-v3"
# SELECT_ENV = "ToyExample-V0"
# SELECT_ENV = "CescoDrive-V1"
# SELECT_ENV = "GraphDrive-Easy"
SELECT_ENV = "GridDrive-Hard"

CONFIG = APPO_DEFAULT_CONFIG.copy()
CONFIG.update({
	# "model": {
	# 	"custom_model": "adaptive_multihead_network",
	# },
	# "lambda": .95, # GAE(lambda) parameter. Taking lambda < 1 introduces bias only when the value function is inaccurate.
	# "clip_param": 0.2, # PPO surrogate loss options; default is 0.4. The higher it is, the higher the chances of catastrophic forgetting.
	# "batch_mode": "complete_episodes", # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes', otherwise it can also be 'truncate_episodes'.
	# "vtrace": False, # Formula for computing the advantages: batch_mode==complete_episodes implies vtrace==False, thus gae==True.
	"rollout_fragment_length": 2**3, # Number of transitions per batch in the experience buffer
	"train_batch_size": 2**9, # Number of transitions per train-batch
	"replay_proportion": 2, # Set a p>0 to enable experience replay. Saved samples will be replayed with a p:1 proportion to new data samples.
	"replay_buffer_num_slots": 2**13, # Maximum number of batches stored in the experience buffer. Every batch has size 'rollout_fragment_length' (default is 50).
})

####################################################################################
####################################################################################

ray.shutdown()
ray.init(ignore_reinit_error=True)

# Configure RLlib to train a policy using the “Taxi-v3” environment and a PPO optimizer
agent = APPOTrainer(CONFIG, env=SELECT_ENV)

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
	last_time = time.time()
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
	print(f'{n+1:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}, len mean: {result["episode_len_mean"]:8.4f}, steps: {result["info"]["num_steps_trained"]:8.4f}, train ratio: {(result["info"]["num_steps_trained"]/result["info"]["num_steps_sampled"]):8.4f}, seconds: {time.time()-last_time}')
	# print(f'Checkpoint saved to {file_name}')

