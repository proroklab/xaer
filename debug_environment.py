import gym
from environments import *

env = GraphDriveHard()
# env = CescoDriveV0()

def run_one_episode (env):
	env.reset()
	sum_reward = 0
	done = False
	while not done:
		action = env.action_space.sample()
		state, reward, done, info = env.step(action)
		sum_reward += reward
		env.render()
	return sum_reward

sum_reward = run_one_episode(env)