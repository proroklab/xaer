# -*- coding: utf-8 -*-
from runstats import Statistics, Regression

class RunningStats(object):
	
	def __init__(self, window_size=None):
		if window_size:
			assert window_size > 1, 'window_size must be greater than 1'
		self.window_size = window_size
		self.running_stats = Statistics()
		
	def push(self, x):
		if self.window_size and len(self.running_stats) >= self.window_size:
			self.running_stats *= (self.window_size-1)/self.window_size
		self.running_stats.push(x)

	@property
	def mean(self):
		return self.running_stats.mean()
	
	@property
	def std(self):
		return self.running_stats.stddev()

	@property
	def var(self):
		return self.running_stats.variance()
	