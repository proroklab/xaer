# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
import multiprocessing
import json
import shutil
import ray

from ray.rllib.agents.dqn.dqn import DQNTrainer, DEFAULT_CONFIG as DQN_DEFAULT_CONFIG
from environments import *

# SELECT_ENV = "Taxi-v3"
# SELECT_ENV = "ToyExample-V0"
SELECT_ENV = "GridDrive-Hard"

CONFIG = DQN_DEFAULT_CONFIG.copy()
CONFIG.update({
	"model": {
		"custom_model": "adaptive_multihead_network",
	},
	"rollout_fragment_length": 1,
	"train_batch_size": 256,
	# "learning_starts": 1500,
	"grad_clip": None,
	# "framework": "torch",
	'buffer_size': 50000, # Size of the experience buffer. Default 50000
	"batch_mode": "complete_episodes", # For some clustering schemes (e.g. extrinsic_reward, moving_best_extrinsic_reward, etc..) it has to be equal to 'complete_episodes', otherwise it can also be 'truncate_episodes'.
})

####################################################################################
####################################################################################

from xarl.models.dqn import TFAdaptiveMultiHeadDQN
from ray.rllib.models import ModelCatalog
# Register the models to use.
ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadDQN)
CONFIG["model"] = {
	"custom_model": "adaptive_multihead_network",
}

####################################################################################
####################################################################################

ray.shutdown()
ray.init(ignore_reinit_error=True)

# Configure RLlib to train a policy using the “Taxi-v3” environment and a PPO optimizer
# agent = DDPGTrainer(CONFIG, env=SELECT_ENV)
agent = DQNTrainer(CONFIG, env=SELECT_ENV)

# Inspect the trained policy and model, to see the results of training in detail
policy = agent.get_policy()
model = policy.model
print(model.q_value_head.summary())
print(model.heads_model.summary())

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

