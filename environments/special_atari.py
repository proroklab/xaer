from gym.envs.atari.atari_env import AtariEnv
import numpy as np

class SpecialAtariEnv(AtariEnv):

	def reset(self):
		obs = super().reset()
		self.lives = self.ale.lives()
		return obs
	
	def step(self, a):
		old_lives = self.lives
		old_ram = self._get_ram()
		state, reward, terminal, info_dict = super().step(a)
		new_ram = self._get_ram()
		new_lives = self.ale.lives()

		lost_lives = old_lives-new_lives
		explanation_list = [np.array2string(new_ram-old_ram) if reward != 0 or lost_lives != 0 else 'no_reward']
		if reward != 0:
			explanation_list.append('reward')
		if lost_lives != 0:
			explanation_list.append('lost_lives')

		info_dict['explanation'] = explanation_list
		self.lives = new_lives
		# print(reward, terminal, delta)

		return state, reward, terminal, info_dict