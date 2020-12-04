from setuptools import setup

with open("requirements.txt", 'r') as f:
	requirements = f.readlines()

setup(
	name='xarl',
	version='1.0',
	description='A plugin for RLlib, providing some extra explanation-aware algorithms.',
	url='https://www.unibo.it/sitoweb/francesco.sovrano2/en',
	author='Francesco Sovrano',
	author_email='cesco.sovrano@gmail.com',
	license='MIT',
	packages=['xarl'],
	# zip_safe=False,
	install_requires=requirements, #external packages as dependencies
	python_requires='>=3.6',
)
