# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
os.environ["TUNE_RESULT_DIR"] = 'tmp/ray_results'
import json
import ray
from ray.rllib.models import ModelCatalog
from xarl.utils.workflow import train

from environments import *

def get_algorithm_by_name(alg_name):
	# DQN
	if alg_name == 'dqn':
		from ray.rllib.agents.dqn.dqn import DQNTrainer, DEFAULT_CONFIG as DQN_DEFAULT_CONFIG
		from xarl.models.dqn import TFAdaptiveMultiHeadDQN
		ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadDQN)
		return DQN_DEFAULT_CONFIG.copy(), DQNTrainer
	if alg_name == 'xadqn':
		from xarl.agents.xadqn import XADQNTrainer, XADQN_DEFAULT_CONFIG
		from xarl.models.dqn import TFAdaptiveMultiHeadDQN
		ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadDQN)
		return XADQN_DEFAULT_CONFIG.copy(), XADQNTrainer
	# DDPG
	if alg_name == 'ddpg':
		from xarl.models.ddpg import TFAdaptiveMultiHeadDDPG
		from ray.rllib.agents.ddpg.ddpg import DDPGTrainer, DEFAULT_CONFIG as DDPG_DEFAULT_CONFIG
		ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadDDPG)
		return DDPG_DEFAULT_CONFIG.copy(), DDPGTrainer
	if alg_name == 'xaddpg':
		from xarl.models.ddpg import TFAdaptiveMultiHeadDDPG
		from xarl.agents.xaddpg import XADDPGTrainer, XADDPG_DEFAULT_CONFIG
		ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadDDPG)
		return XADDPG_DEFAULT_CONFIG.copy(), XADDPGTrainer
	# TD3
	if alg_name == 'td3':
		from xarl.models.ddpg import TFAdaptiveMultiHeadDDPG
		from ray.rllib.agents.ddpg.td3 import TD3Trainer, TD3_DEFAULT_CONFIG
		ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadDDPG)
		return TD3_DEFAULT_CONFIG.copy(), TD3Trainer
	if alg_name == 'xatd3':
		from xarl.models.ddpg import TFAdaptiveMultiHeadDDPG
		from xarl.agents.xaddpg import XATD3Trainer, XATD3_DEFAULT_CONFIG
		ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadDDPG)
		return XATD3_DEFAULT_CONFIG.copy(), XATD3Trainer
	# SAC
	if alg_name == 'sac':
		from ray.rllib.agents.sac.sac import SACTrainer, DEFAULT_CONFIG as SAC_DEFAULT_CONFIG
		return SAC_DEFAULT_CONFIG.copy(), SACTrainer
	if alg_name == 'xasac':
		from xarl.agents.xasac import XASACTrainer, XASAC_DEFAULT_CONFIG
		return XASAC_DEFAULT_CONFIG.copy(), XASACTrainer
	# PPO
	if alg_name in ['appo','ppo']:
		from ray.rllib.agents.ppo.appo import APPOTrainer, DEFAULT_CONFIG as APPO_DEFAULT_CONFIG
		from xarl.models.appo import TFAdaptiveMultiHeadNet
		ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadNet)
		return APPO_DEFAULT_CONFIG.copy(), APPOTrainer
	if alg_name == 'xappo':
		from xarl.agents.xappo import XAPPOTrainer, XAPPO_DEFAULT_CONFIG
		from xarl.models.appo import TFAdaptiveMultiHeadNet
		ModelCatalog.register_custom_model("adaptive_multihead_network", TFAdaptiveMultiHeadNet)
		return XAPPO_DEFAULT_CONFIG.copy(), XAPPOTrainer

import sys
CONFIG, TRAINER = get_algorithm_by_name(sys.argv[1])
ENVIRONMENT = sys.argv[2]
TEST_EVERY_N_STEP = int(float(sys.argv[3]))
STOP_TRAINING_AFTER_N_STEP = int(float(sys.argv[4]))
if len(sys.argv) > 5:
	OPTIONS = json.loads(' '.join(sys.argv[5:]))
	CONFIG.update(OPTIONS)
CONFIG["callbacks"] = CustomEnvironmentCallbacks
print(CONFIG)

####################################################################################
####################################################################################

ray.shutdown()
ray.init(ignore_reinit_error=True)

train(TRAINER, CONFIG, ENVIRONMENT, test_every_n_step=TEST_EVERY_N_STEP, stop_training_after_n_step=STOP_TRAINING_AFTER_N_STEP)
