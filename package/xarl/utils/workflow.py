# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
import shutil
import time
import xarl.utils.plot_lib as plt
import zipfile
import sys
from io import StringIO
from contextlib import closing
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind, is_atari
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
import numpy as np

def test(tester_class, config, environment_class, checkpoint, save_gif=True, delete_screens_after_making_gif=True, compress_gif=True, n_episodes=5):
	"""Tests and renders a previously trained model"""
	# test_config = config.copy()
	# test_config['explore'] = False
	agent = tester_class(config, env=environment_class)
	if checkpoint is None:
		raise ValueError(f"A previously trained checkpoint must be provided for algorithm {alg}")
	agent.restore(checkpoint)

	checkpoint_directory = os.path.dirname(checkpoint)
	env = agent.env_creator(config["env_config"])
	# Atari wrapper
	if is_atari(env) and not config.get("custom_preprocessor") and config.get("preprocessor_pref","deepmind") == "deepmind":
		# Deprecated way of framestacking is used.
		framestack = config.get("framestack") is True
		# framestacking via trajectory view API is enabled.
		num_framestacks = config.get("num_framestacks", 0)

		# Trajectory view API is on and num_framestacks=auto:
		# Only stack traj. view based if old
		# `framestack=[invalid value]`.
		if num_framestacks == "auto":
			if framestack == DEPRECATED_VALUE:
				config["num_framestacks"] = num_framestacks = 4
			else:
				config["num_framestacks"] = num_framestacks = 0
		framestack_traj_view = num_framestacks > 1
		env = wrap_deepmind(
			env,
			# dim=config.get("dim"),
			framestack=framestack,
			framestack_via_traj_view_api=framestack_traj_view
		)

	render_modes = env.metadata['render.modes']
	env.seed(config["seed"])
	def print_screen(screens_directory, step):
		filename = os.path.join(screens_directory, f'frame{step}.jpg')
		if 'rgb_array' in render_modes:
			plt.rgb_array_image(
				env.render(mode='rgb_array'), 
				filename
			)
		elif 'ansi' in render_modes:
			plt.ascii_image(
				env.render(mode='ansi'), 
				filename
			)
		elif 'ascii' in render_modes:
			plt.ascii_image(
				env.render(mode='ascii'), 
				filename
			)
		elif 'human' in render_modes:
			old_stdout = sys.stdout
			sys.stdout = StringIO()
			env.render(mode='human')
			with closing(sys.stdout):
				plt.ascii_image(
					sys.stdout.getvalue(), 
					filename
				)
			sys.stdout = old_stdout
		else:
			raise Exception(f"No compatible render mode (rgb_array,ansi,ascii,human) in {render_modes}.")
		return filename

	for episode_id in range(n_episodes):
		episode_directory = os.path.join(checkpoint_directory, f'episode_{episode_id}')
		os.mkdir(episode_directory)
		screens_directory = os.path.join(episode_directory, 'screen')
		os.mkdir(screens_directory)
		log_list = []
		sum_reward = 0
		step = 0
		done = False
		state = np.squeeze(env.reset())
		file_list = [print_screen(screens_directory, step)]
		while not done:
			step += 1
			# action = env.action_space.sample()
			action = agent.compute_action(state, full_fetch=True, explore=False)
			state, reward, done, info = env.step(action[0])
			state = np.squeeze(state)
			sum_reward += reward
			file_list.append(print_screen(screens_directory, step))
			log_list.append(', '.join([
				f'step: {step}',
				f'reward: {reward}',
				f'done: {done}',
				f'info: {info}',
				f'action: {action}',
				f'state: {state}',
				f'\n\n',
			]))
		with open(episode_directory + f'/episode_{step}_{sum_reward}.log', 'w') as f:
			f.writelines(log_list)
		if save_gif:
			gif_filename = os.path.join(episode_directory, 'episode.gif')
			plt.make_gif(file_list=file_list, gif_path=gif_filename)
			# Delete screens, to save memory
			if delete_screens_after_making_gif:
				shutil.rmtree(screens_directory, ignore_errors=True)
			# Zip GIF, to save memory
			if compress_gif:
				with zipfile.ZipFile(gif_filename+'.zip', mode='w', compression=zipfile.ZIP_DEFLATED) as z:
					z.write(gif_filename,'episode.gif')
				# Remove unzipped GIF
				os.remove(gif_filename)

def train(trainer_class, config, environment_class, test_every_n_step=None, stop_training_after_n_step=None, log=True):
	# Configure RLlib to train a policy using the given environment and trainer
	agent = trainer_class(config, env=environment_class)
	# # Inspect the trained policy and model, to see the results of training in detail
	# policy = agent.get_policy()
	# model = policy.model
	# if hasattr(model, 'base_model'):
	# 	print(model.base_model.summary())
	# if hasattr(model, 'q_value_head'):
	# 	print(model.q_value_head.summary())
	# if hasattr(model, 'heads_model'):
	# 	print(model.heads_model.summary())
	# Start training
	n = 0
	sample_steps = 0
	if stop_training_after_n_step is None:
		stop_training_after_n_step = float('inf')
	check_steps = test_every_n_step if test_every_n_step is not None else float('inf')
	def save_checkpoint():
		checkpoint = agent.save()
		print(f'Checkpoint saved in {checkpoint}')
		print(f'Testing..')
		try:
			test(trainer_class, config, environment_class, checkpoint)
		except Exception as e:
			print(e)
	while sample_steps < stop_training_after_n_step:
		n += 1
		last_time = time.time()
		result = agent.train()
		train_steps = result["info"]["num_steps_trained"]
		sample_steps = result["info"]["num_steps_sampled"]
		episode = {
			'n': n, 
			'episode_reward_min': result['episode_reward_min'], 
			'episode_reward_mean': result['episode_reward_mean'], 
			'episode_reward_max': result['episode_reward_max'],  
			'episode_len_mean': result['episode_len_mean']
		}
		if log:
			print(', '.join([
				f'iteration: {n+1}',
				f'episode_reward (min/mean/max): {result["episode_reward_min"]:.2f}/{result["episode_reward_mean"]:.2f}/{result["episode_reward_max"]:.2f}',
				f'episode_len_mean: {result["episode_len_mean"]:.2f}',
				f'steps_trained: {train_steps}',
				f'steps_sampled: {sample_steps}',
				f'train_ratio: {(train_steps/sample_steps):.2f}',
				f'seconds: {time.time()-last_time:.2f}'
			]))
		if sample_steps>=check_steps or sample_steps>=stop_training_after_n_step:
			check_steps += test_every_n_step
			save_checkpoint()
