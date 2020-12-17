# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
import multiprocessing
import json
import shutil
import ray

from environments import *

def get_algorithm_by_name(alg_name):
	if alg_name == 'dqn':
		from ray.rllib.agents.dqn.dqn import DQNTrainer, DEFAULT_CONFIG as DQN_DEFAULT_CONFIG
		from xarl.agents.xadqn import DQNTrainer as DQNTrainer_ExplainedVar
		return DQN_DEFAULT_CONFIG.copy(), DQNTrainer_ExplainedVar
	if alg_name == 'xadqn':
		from xarl.agents.xadqn import XADQNTrainer, XADQN_DEFAULT_CONFIG
		return XADQN_DEFAULT_CONFIG.copy(), XADQNTrainer
	if alg_name == 'ddpg':
		from ray.rllib.agents.ddpg.ddpg import DDPGTrainer, DEFAULT_CONFIG as DDPG_DEFAULT_CONFIG
		from xarl.agents.xaddpg import DDPGTrainer as DDPGTrainer_ExplainedVar
		return DDPG_DEFAULT_CONFIG.copy(), DDPGTrainer_ExplainedVar
	if alg_name == 'xaddpg':
		from xarl.agents.xaddpg import XADDPGTrainer, XADDPG_DEFAULT_CONFIG
		return XADDPG_DEFAULT_CONFIG.copy(), XADDPGTrainer
	if alg_name == 'appo':
		from ray.rllib.agents.ppo.appo import APPOTrainer, DEFAULT_CONFIG as APPO_DEFAULT_CONFIG
		return APPO_DEFAULT_CONFIG.copy(), APPOTrainer
	if alg_name == 'xappo':
		from xarl.agents.xappo import XAPPOTrainer, XAPPO_DEFAULT_CONFIG
		return XAPPO_DEFAULT_CONFIG.copy(), XAPPOTrainer

import sys
CONFIG, TRAINER = get_algorithm_by_name(sys.argv[1])
ENVIRONMENT = sys.argv[2]
if len(sys.argv) > 3:
	OPTIONS = json.loads(' '.join(sys.argv[3:]))
	CONFIG.update(OPTIONS)
print(CONFIG)

####################################################################################
####################################################################################

ray.shutdown()
ray.init(ignore_reinit_error=True)

# Configure RLlib to train a policy using the given environment and trainer
agent = TRAINER(CONFIG, env=ENVIRONMENT)

n = 0
while True:
	n += 1
	result = agent.train()
	episode = {
		'n': n, 
		'episode_reward_min': result['episode_reward_min'], 
		'episode_reward_mean': result['episode_reward_mean'], 
		'episode_reward_max': result['episode_reward_max'],  
		'episode_len_mean': result['episode_len_mean']
	}
	print(f'{n+1:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}, len mean: {result["episode_len_mean"]:8.4f}')
	# file_name = agent.save(checkpoint_root)
	# print(f'Checkpoint saved to {file_name}')

