# Read this guide for how to use this script: https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70
import os
import shutil
import time
import xarl.utils.plot_lib as plt
import zipfile

def test(tester_class, config, environment_class, checkpoint, save_gif=True, delete_screens_after_making_gif=True, compress_gif=True):
	"""Tests and renders a previously trained model"""
	test_config = config.copy()
	test_config['explore'] = False
	agent = tester_class(test_config, env=environment_class)
	if checkpoint is None:
		raise ValueError(f"A previously trained checkpoint must be provided for algorithm {alg}")
	agent.restore(checkpoint)
	
	episode_directory = os.path.dirname(checkpoint)
	screens_directory = os.path.join(episode_directory,'screen')
	# os.mkdir(episode_directory)
	os.mkdir(screens_directory)
	file_list = []

	env = agent.env_creator(test_config["env_config"])
	sum_reward = 0
	step = 0
	done = False
	state = env.reset()
	while not done:
		# action = env.action_space.sample()
		action = agent.compute_action(state)
		state, reward, done, info = env.step(action)
		sum_reward += reward
		filename = os.path.join(screens_directory, f'frame{step}.jpg')
		plt.rgb_array_image(
			env.render(mode='rgb_array'), 
			filename
		)
		file_list.append(filename)
		step += 1
	if save_gif:
		gif_filename = os.path.join(episode_directory, f'episode_{sum_reward}.gif')
		plt.make_gif(file_list=file_list, gif_path=gif_filename)
		# Delete screens, to save memory
		if delete_screens_after_making_gif:
			shutil.rmtree(screens_directory, ignore_errors=True)
		# Zip GIF, to save memory
		if compress_gif:
			with zipfile.ZipFile(gif_filename+'.zip', mode='w', compression=zipfile.ZIP_DEFLATED) as zip:
				zip.write(gif_filename)
			# Remove unzipped GIF
			os.remove(gif_filename)

def train(trainer_class, config, environment_class, test_every_n_step=None, stop_training_after_n_step=None):
	# Configure RLlib to train a policy using the given environment and trainer
	agent = trainer_class(config, env=environment_class)
	# Inspect the trained policy and model, to see the results of training in detail
	policy = agent.get_policy()
	model = policy.model
	if hasattr(model, 'base_model'):
		print(model.base_model.summary())
	if hasattr(model, 'q_value_head'):
		print(model.q_value_head.summary())
	if hasattr(model, 'heads_model'):
		print(model.heads_model.summary())
	# Start training
	n = 0
	while stop_training_after_n_step is None or n < stop_training_after_n_step:
		n += 1
		last_time = time.time()
		result = agent.train()
		episode = {
			'n': n, 
			'episode_reward_min': result['episode_reward_min'], 
			'episode_reward_mean': result['episode_reward_mean'], 
			'episode_reward_max': result['episode_reward_max'],  
			'episode_len_mean': result['episode_len_mean']
		}
		print(f'{n+1:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}, len mean: {result["episode_len_mean"]:8.4f}, steps: {result["info"]["num_steps_trained"]:8.4f}, train ratio: {(result["info"]["num_steps_trained"]/result["info"]["num_steps_sampled"]):8.4f}, seconds: {time.time()-last_time}')
		# file_name = agent.save(checkpoint_root)
		# print(f'Checkpoint saved to {file_name}')
		if test_every_n_step is not None and n%test_every_n_step==0:
			checkpoint = agent.save()
			print(f'Checkpoint saved in {checkpoint}')
			print(f'Testing..')
			test(trainer_class, config, environment_class, checkpoint)
