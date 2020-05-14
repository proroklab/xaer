# -*- coding: utf-8 -*-
import time
from multiprocessing import Process, SimpleQueue as Queue
# from multiprocessing import Queue
# from threading import Thread as Process
import numpy as np
import random
from utils.statistics import Statistics

import options
flags = options.get()

class Environment(object):
	state_scaler = 1

	@staticmethod
	def create_environment(env_type, env_id=0, training=True, group_id=0):
		if env_type == 'CarController':
			from environment.car_controller.car_controller_wrapper import CarControllerGameWrapper
			return Environment(env_id, CarControllerGameWrapper)
		else:
			from environment.openai_gym.openai_gym_wrapper import GymGameWrapper
			Environment.state_scaler = GymGameWrapper.state_scaler
			return Environment(env_id, GymGameWrapper, {'game': env_type})

	@staticmethod
	def __game_worker(input_queue, output_queue, game_wrapper, config_dict):
		def get_observation_dict(game):
			observation_dict = {
				'state': game.last_state,
				'reward': game.last_reward,
				'step': game.step,
				'is_over': game.is_over,
				'statistics': game.get_statistics(),
			}
			if config_dict.get('get_screen',False):
				observation_dict['screen'] = game.get_screen()
				observation_dict['info'] = game.get_info()
			return observation_dict
		game = game_wrapper(config_dict)
		game.reset()
		observation_dict = get_observation_dict(game)
		output_queue.put(observation_dict) # step = 0
		while not game.is_over:
			action = input_queue.get()
			# print(action)
			if action is None:   # If you send `None`, the thread will exit.
				return
			game.process(action)
			observation_dict = get_observation_dict(game)
			output_queue.put(observation_dict) # step > 0

	def __init__(self, id, game_wrapper, config_dict={}):
		self.id = id
		self.__config_dict = config_dict
		self.__game_wrapper = game_wrapper
		self.__game_thread = None

		# Statistics
		self.__episode_statistics = Statistics(flags.episode_count_for_evaluation)
		
		tmp_game = game_wrapper(config_dict)
		self.__state_shape = tmp_game.get_state_shape()
		self.__action_shape = tmp_game.get_action_shape()
		self.__has_masked_actions = tmp_game.has_masked_actions()

	def get_state_shape(self):
		return self.__state_shape

	def get_action_shape(self):
		return self.__action_shape

	def stop(self):
		if self.__game_thread is not None:
			# print('Closing..')
			self.__input_queue.put(None)
			self.__game_thread.join()
			self.__game_thread.terminate()
			self.__game_thread = None
			# print('Closed')

	def reset(self, data_id=None, get_screen=False):
		self.stop()
		self.__config_dict['get_screen'] = get_screen
		# print('Starting..')
		self.__input_queue = Queue()
		self.__output_queue = Queue()
		self.__game_thread = Process(
			target=self.__game_worker, 
			args=(self.__input_queue, self.__output_queue, self.__game_wrapper, self.__config_dict)
		)
		# self.__game_thread.daemon = True
		self.__game_thread.start()
		time.sleep(0.1)
		self.last_observation = self.__output_queue.get()
		self.last_state = self.last_observation['state']
		self.last_reward = 0
		self.last_action = None
		self.step = 0
		#print(self.id, self.step)
		return self.last_state

	def process(self, action_vector):
		self.__input_queue.put(action_vector)
		if self.__game_thread.is_alive():
			self.last_observation = self.__output_queue.get()
			is_terminal = self.last_observation['is_over']
			self.last_state = self.last_observation['state']
			self.last_reward = self.last_observation['reward'] if not is_terminal else 1
			self.last_action = action_vector
			if is_terminal:
				self.__episode_statistics.add(self.last_observation['statistics'])
		else:
			is_terminal = True
		# complete step
		self.step += 1
		return self.last_state, self.last_reward, is_terminal, None
		
	def sample_random_action(self):
		result = []
		for action_shape in self.get_action_shape():
			if len(action_shape) > 1:
				count, size = action_shape
			else:
				count = action_shape[0]
				size = 0
			if size > 0: # categorical sampling
				samples = (np.random.rand(count)*size).astype(np.uint8)
				result.append(samples if count > 1 else samples[0])
			else: # uniform sampling
				result.append([2*random.random()-1 for _ in range(count)])
		return result
		
	def get_test_result(self):
		return None
		
	def get_dataset_size(self):
		return flags.episode_count_for_evaluation
		
	def get_screen_shape(self):
		return self.get_state_shape()
	
	def get_info(self):
		return self.last_observation.get('info',{})
	
	def get_screen(self):
		return self.last_observation.get('screen',{})
	
	def get_statistics(self):
		return self.__episode_statistics.get()
	
	def has_masked_actions(self):
		return self.__has_masked_actions

	def evaluate_test_results(self, test_result_file):
		pass

