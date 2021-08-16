from gym.envs.atari.atari_env import AtariEnv
import numpy as np

class SpecialAtariEnv(AtariEnv):

	# def reset(self):
	# 	obs = super().reset()
	# 	self.last_delta = None
	# 	return obs
	
	def step(self, a):
		old_ram = self._get_ram()
		state, reward, terminal, info_dict = super().step(a)
		new_ram = self._get_ram()

		delta = new_ram-old_ram
		info_dict['explanation'] = np.array2string(delta) if reward != 0 else 'no_reward'
		# print(reward, terminal, delta)

		return state, reward, terminal, info_dict