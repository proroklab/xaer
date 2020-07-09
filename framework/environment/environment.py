# -*- coding: utf-8 -*-
import time
from multiprocessing import Process, Queue
# from multiprocessing import Queue
# from threading import Thread as Process
import numpy as np
import random
from utils.statistics import Statistics

import options
flags = options.get()

DEBUG = False

def get_timed_queue(q, timeout=None, qid=None):
	return q.get(timeout=timeout)
	# try:
	# 	return q.get(timeout=timeout)
	# except:
	# 	if DEBUG:
	# 		print("restarting", qid)
	# 	time.sleep(0.1)
	# 	return get_timed_queue(q,timeout,qid)

def put_timed_queue(q, v, timeout=None, qid=None):
	q.put(v, timeout=timeout)
	# try:
	# 	q.put(v, timeout=timeout)
	# except:
	# 	if DEBUG:
	# 		print("restarting", qid)
	# 	time.sleep(0.1)
	# 	put_timed_queue(q,v,timeout,qid)

class Environment(object):
	state_scaler = 1

	@staticmethod
	def create_environment(env_type, env_id=0, training=True, group_id=0):
		if env_type.startswith('CarController'):
			import environment.car_controller as car_controller
			return Environment(f'{group_id}.{env_id}', eval(f'car_controller.{env_type}'))
		else:
			from environment.openai_gym.openai_gym_wrapper import GymGameWrapper
			Environment.state_scaler = GymGameWrapper.state_scaler
			return Environment(f'{group_id}.{env_id}', GymGameWrapper, {'game': env_type})

	@staticmethod
	def __game_worker(input_queue, output_queue, game_wrapper, config_dict):
		def get_observation_dict(game):
			observation_dict = {
				'new_state': game.last_state,
				'reward': game.last_reward,
				'reward_type': game.last_reward_type,
				'action_mask': game.last_action_mask,
				# 'step': game.step,
				'is_over': game.is_over,
				'statistics': game.get_statistics(),
			}
			if config_dict.get('get_screen',False):
				observation_dict['screen'] = game.get_screen()
				observation_dict['info'] = game.get_info()
			return observation_dict
		game = game_wrapper(config_dict)
		game.reset()
		put_timed_queue(output_queue, get_observation_dict(game), qid=config_dict['id']) # step == 0
		while True:
			action = get_timed_queue(input_queue, qid=config_dict['id'])
			if action is None or game.is_over:   # If you send `None`, the thread will reset.
				game.reset()
			else:
				game.process(action)
			put_timed_queue(output_queue, get_observation_dict(game), qid=config_dict['id']) # step > 0

	def __init__(self, gid, game_wrapper, config_dict=None):
		self.id = gid
		self.__config_dict = config_dict if config_dict else {}
		self.__config_dict['id'] = self.id
		self.__game_wrapper = game_wrapper
		self.__game_thread = None
		# Statistics
		self.__episode_statistics = Statistics(flags.episode_count_for_evaluation)
		tmp_game = game_wrapper(config_dict)
		self.__state_shape = tmp_game.get_state_shape()
		self.__action_shape = tmp_game.get_action_shape()
		self.__has_masked_actions = tmp_game.has_masked_actions()
		# Game Process
		self.__game_thread = None

	def __start_thread(self):
		if self.__game_thread is not None:
			self.__close_thread()
		if DEBUG:
			print('Starting thread', self.id)
		self.__input_queue = Queue()
		self.__output_queue = Queue()
		self.__game_thread = Process(
			target=self.__game_worker, 
			args=(self.__input_queue, self.__output_queue, self.__game_wrapper, self.__config_dict)
		)
		self.__game_thread.daemon = True
		self.__game_thread.start()

	def __close_thread(self):
		if self.__game_thread is None:
			return
		if DEBUG:
			print('Closing thread', self.id)
		self.__game_thread.kill()
		while self.__game_thread.is_alive():
			time.sleep(1e-2)
		self.__game_thread.close()
		self.__input_queue.close()
		self.__output_queue.close()
		if DEBUG:
			print('Thread', self.id, 'closed')
		self.__game_thread = None

	def get_state_shape(self):
		return self.__state_shape

	def get_action_shape(self):
		return self.__action_shape

	def stop(self):
		if DEBUG:
			print('Stopping game', self.id)
		self.__close_thread()

	def reset(self, data_id=None, get_screen=False):
		if self.__game_thread is None or get_screen != self.__config_dict.get('get_screen',False): # start a new thread
			self.__config_dict['get_screen'] = get_screen
			if DEBUG:
				print('Change config', self.__config_dict)
			self.__start_thread()
		else: # reuse the current thread
			put_timed_queue(self.__input_queue, None, qid=self.id)
		if DEBUG:
			print('Resetting game', self.id)
		self.last_observation = get_timed_queue(self.__output_queue, qid=self.id)
		self.step = 0
		return self.last_observation

	def process(self, action_vector):
		put_timed_queue(self.__input_queue, action_vector, qid=self.id)
		if self.__game_thread.is_alive():
			self.last_observation = get_timed_queue(self.__output_queue, qid=self.id)
			is_terminal = self.last_observation['is_over']
			if is_terminal:
				self.__episode_statistics.add(self.last_observation['statistics'])
		else:
			self.__game_thread = None
			is_terminal = True
		# complete step
		self.step += 1
		return self.last_observation, is_terminal
		
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

