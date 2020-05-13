#cython initializedcheck=False, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# CTS code adapted from https://github.com/mgbellemare/SkipCTS

#===============================================================================
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#===============================================================================

cimport cython
import numpy as np
cimport numpy as np

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.math cimport log, exp
from cpython cimport array

import cv2

# Parameters of the CTS model. For clarity, we take these as constants.
cdef double PRIOR_STAY_PROB = 0.5
cdef double PRIOR_SPLIT_PROB = 0.5
cdef double LOG_PRIOR_STAY_PROB = log(PRIOR_STAY_PROB)
cdef double LOG_PRIOR_SPLIT_PROB = log(1.0 - PRIOR_STAY_PROB)
# Sampling parameter. The maximum number of rejections before we give up and
# sample from the root estimator.
cdef int MAX_SAMPLE_REJECTIONS = 25

cdef double get_prior(char * prior_name, int alphabet_size):
	if prior_name == < char *> 'perks':
		return 1.0 / < double > alphabet_size
	elif prior_name == < char *> 'jeffreys':
		return 0.5
	else:  # use laplace prior
		return 1.0

cdef double log_add(double log_x, double log_y):
	"""Given log x and log y, returns log(x + y)."""
	# Swap variables so log_y is larger.
	if log_x > log_y:
		log_x, log_y = log_y, log_x

	cdef double delta = log_y - log_x
	return log(1 + exp(delta)) + log_x if delta <= 50.0 else log_y

cdef struct EstimatorStruct:
	unsigned int alphabet_size
	double count_total
	double * counts

cdef EstimatorStruct * make_estimator(CTSStruct * model):
	cdef EstimatorStruct * e = < EstimatorStruct *> PyMem_Malloc(sizeof(EstimatorStruct))

	e[0].counts = < double *> PyMem_Malloc(model[0].alphabet_size * sizeof(double))
	cdef unsigned int i
	for i in range(model[0].alphabet_size):
		e[0].counts[i] = model[0].symbol_prior

	e[0].count_total = model[0].alphabet_size * model[0].symbol_prior
	e[0].alphabet_size = model[0].alphabet_size
	return e

cdef void free_estimator(EstimatorStruct * e):
	PyMem_Free(e[0].counts)
	PyMem_Free(e)

cdef double estimator_prob(EstimatorStruct * e, int symbol):
	cdef EstimatorStruct estimator = e[0]
	return e[0].counts[symbol] / e[0].count_total

cdef double estimator_update(EstimatorStruct * e, int symbol):
	cdef double prob = estimator_prob(e, symbol)
	cdef double log_prob = log(prob)
	e[0].counts[symbol] = e[0].counts[symbol] + 1.0
	e[0].count_total += 1.0
	return log_prob

cdef estimator_get_state(EstimatorStruct * ptr):
	return ptr[0].alphabet_size, ptr[0].count_total, [
		ptr[0].counts[i] for i in range(ptr[0].alphabet_size)]

cdef estimator_set_state(EstimatorStruct * ptr, state):
	ptr[0].alphabet_size, ptr[0].count_total = state[:2]
	cdef unsigned int i
	for i in range(ptr[0].alphabet_size):
		ptr[0].counts[i] = state[2][i]
			
cdef struct CTSNodeStruct:
	double _log_stay_prob
	double _log_split_prob
	CTSStruct * _model
	EstimatorStruct * estimator
	CTSNodeStruct * _children

cdef CTSNodeStruct * make_cts_node(CTSStruct * model):
	cdef CTSNodeStruct * node = < CTSNodeStruct *> PyMem_Malloc(sizeof(CTSNodeStruct))
	node[0]._children = NULL
	node[0].estimator = make_estimator(model)
	node[0]._model = model
	
	node[0]._log_stay_prob = LOG_PRIOR_STAY_PROB
	node[0]._log_split_prob = LOG_PRIOR_SPLIT_PROB

	return node

cdef void free_cts_node(CTSNodeStruct * node):
	free_estimator(node[0].estimator)
	PyMem_Free(node[0]._children)

cdef double node_update(CTSNodeStruct * node, int[:] context, int symbol):
	lp_estimator = estimator_update(node[0].estimator, symbol)

	# If not a leaf node, recurse, creating nodes as needed.
	cdef CTSNodeStruct * child
	cdef double lp_child
	cdef double lp_node
	if context.shape[0] > 0:
		child = node_get_child(node, context[context.shape[0] - 1])
		lp_child = node_update(child, context[:context.shape[0] - 1], symbol)
		lp_node = node_mix_prediction(node, lp_estimator, lp_child)

		node_update_switching_weights(node, lp_estimator, lp_child)

		return lp_node
	else:
		node[0]._log_stay_prob = 0.0
		return lp_estimator

cdef double node_log_prob(CTSNodeStruct * node, int[:] context, int symbol):
	cdef double lp_estimator = log(estimator_prob(node[0].estimator, symbol))
	cdef CTSNodeStruct * child
	
	if context.shape[0] > 0:
		child = node_get_child(node, context[context.shape[0] - 1])
		lp_child = node_log_prob(child, context[:context.shape[0] - 1], symbol)

		return node_mix_prediction(node, lp_estimator, lp_child)
	else:
		return lp_estimator

cdef CTSNodeStruct * node_get_child(CTSNodeStruct * node, int symbol):
	if node[0]._children == NULL:
		node[0]._children = < CTSNodeStruct *> PyMem_Malloc(node._model[0].alphabet_size * sizeof(CTSNodeStruct))
		for i in range(node._model[0].alphabet_size):
			node[0]._children[i] = make_cts_node(node._model)[0]

	return & node[0]._children[symbol]

cdef double node_mix_prediction(CTSNodeStruct * node, double lp_estimator, double lp_child):
	cdef double numerator = log_add(lp_estimator + node[0]._log_stay_prob,
								 lp_child + node[0]._log_split_prob)
	cdef double denominator = log_add(node[0]._log_stay_prob,
								   node[0]._log_split_prob)

	return numerator - denominator

cdef void node_update_switching_weights(CTSNodeStruct * node, double lp_estimator, double lp_child):
	cdef double log_alpha = node[0]._model[0].log_alpha
	cdef double log_1_minus_alpha = node[0]._model[0].log_1_minus_alpha

	# Avoid numerical issues with alpha = 1. This reverts to straight up
	# weighting.
	if log_1_minus_alpha == 0:
		node[0]._log_stay_prob += lp_estimator
		node[0]._log_split_prob += lp_child

	else:
		node[0]._log_stay_prob = log_add(log_1_minus_alpha
											   +lp_estimator
											   +node[0]._log_stay_prob,
											   log_alpha
											   +lp_child
											   +node[0]._log_split_prob)

		node[0]._log_split_prob = log_add(log_1_minus_alpha
												+lp_child
												+node[0]._log_split_prob,
												log_alpha
												+lp_estimator
												+node[0]._log_stay_prob)

cdef node_get_state(CTSNodeStruct * ptr):
	child_states = None
	if ptr[0]._children != NULL:
		child_states = [node_get_state(& ptr[0]._children[i]) for i in range(ptr[0]._model[0].alphabet_size)]
	return ptr[0]._log_stay_prob, ptr[0]._log_split_prob, estimator_get_state(ptr[0].estimator), child_states

cdef node_set_state(CTSNodeStruct * ptr, state):
	ptr[0]._log_stay_prob, ptr[0]._log_split_prob, estimator_state, child_states = state
	estimator_set_state(ptr[0].estimator, estimator_state)
	if child_states is not None:
		if ptr[0]._children == NULL:
			ptr[0]._children = < CTSNodeStruct *> PyMem_Malloc(ptr[0]._model[0].alphabet_size * sizeof(CTSNodeStruct))
			for i in range(ptr[0]._model[0].alphabet_size):
				ptr[0]._children[i] = make_cts_node(ptr[0]._model)[0]

		for i in range(ptr[0]._model[0].alphabet_size):
			node_set_state(& ptr[0]._children[i], child_states[i])

cdef struct CTSStruct:	
	double _time
	unsigned int context_length
	unsigned int alphabet_size
	double log_alpha
	double log_1_minus_alpha
	double symbol_prior
	CTSNodeStruct * _root

cdef CTSStruct * make_cts(int context_length, int max_alphabet_size=256, char * symbol_prior= < char *> 'perks'):
	cdef CTSStruct * cts = < CTSStruct *> PyMem_Malloc(sizeof(CTSStruct))
	# Total number of symbols processed.
	cts[0]._time = 0.0
	cts[0].context_length = context_length		
	cts[0].alphabet_size = max_alphabet_size

	# These are properly set when we call update().
	cts[0].log_alpha, cts[0].log_1_minus_alpha = 0.0, 0.0
	cts[0].symbol_prior = get_prior(symbol_prior, cts[0].alphabet_size) 

	# Create root. This must happen after setting alphabet & symbol prior.
	cts[0]._root = make_cts_node(cts)
	return cts

cdef void free_cts(CTSStruct * cts):
	free_cts_node(cts[0]._root)

cdef double cts_update(CTSStruct * cts, int[:] context, int symbol):
	cts[0]._time += 1.0
	cts[0].log_alpha = log(1.0 / (cts[0]._time + 1.0))
	cts[0].log_1_minus_alpha = log(cts[0]._time / (cts[0]._time + 1.0))

	cdef double log_prob = node_update(cts[0]._root, context, symbol)

	return log_prob

cdef double cts_log_prob(CTSStruct * cts, int[:] context, int symbol):
	# context is assumed to have correct length
	return node_log_prob(cts[0]._root, context, symbol)

cdef class CTS:
	cdef CTSStruct * inner
	
	def __init__(self, context_length, alphabet_size):
		self.inner = make_cts(context_length, alphabet_size)
		
	cpdef double update(self, int[:] context, int symbol):
		return cts_update(self.inner, context, symbol)
		
	cpdef double log_prob(self, int[:] context, int symbol):
		return cts_log_prob(self.inner, context, symbol)

cdef class CTSDensityModel:
	cdef unsigned int height
	cdef unsigned int width
	cdef float beta
	cdef CTSStruct ** cts_factors

	def __init__(self, observation_shape, beta=0.05):
		observation_shape = np.array(observation_shape, dtype=np.uint32)
		self.height = observation_shape[0]
		self.width = observation_shape[1]
		self.beta = beta
		
		self.cts_factors = < CTSStruct **> PyMem_Malloc(sizeof(CTSStruct *) * self.height)
		cdef int i, j
		for i in range(self.height):
			self.cts_factors[i] = < CTSStruct *> PyMem_Malloc(sizeof(CTSStruct) * self.width)
			for j in range(self.width):
				self.cts_factors[i][j] = make_cts(4, max_alphabet_size=8)[0]
				
	def __dealloc__(self):
		pass

	def reduce_observation_alphabet(self, obs):
		obs = np.array(obs)
		if np.issubdtype(obs.dtype, np.floating):
			obs = np.rint(obs) # round to nearest int
			obs = obs.astype(np.uint8)
		if np.issubdtype(obs.dtype, np.integer):
			obs = obs.astype(np.uint8)
		assert np.issubdtype(obs.dtype, np.uint8), 'Observation must be float or int'
		# reduce alphabet size from 2^8 to 2^3
		obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
		obs = (obs / 255) * 7
		obs = np.rint(obs)  # round to nearest int
		#=======================================================================
		# obs = obs.astype(np.uint8)
		#=======================================================================
		return obs

	def update(self, obs):
		obs = self.reduce_observation_alphabet(obs).astype(np.int32)
		
		log_prob, log_recoding_prob = self._update(obs)
		return self.exploration_bonus(log_prob, log_recoding_prob)
	
	cpdef (double, double) _update(self, int[:, :] obs):
		cdef int[:] context = np.zeros(4, np.int32)
		cdef double log_prob = 0.0
		cdef double log_recoding_prob = 0.0
		cdef unsigned int i
		cdef unsigned int j

		for i in range(self.height):
			for j in range(self.width):
				if self.height > 1 and self.width > 1:  # L-shaped context
					context[3] = obs[i, j - 1] if j > 0 else 0
					context[2] = obs[i - 1, j] if i > 0 else 0
					context[1] = obs[i - 1, j - 1] if i > 0 and j > 0 else 0
					context[0] = obs[i - 1, j + 1] if i > 0 and j + 1 < self.width else 0
				elif self.height > 1:
					context[3] = obs[i + 2, j] if i + 2 < self.height else 0
					context[2] = obs[i + 1, j] if i + 1 < self.height else 0
					context[1] = obs[i - 1, j] if i - 1 > 0 else 0
					context[0] = obs[i - 2, j] if i - 2 > 0 else 0
				else:
					context[3] = obs[i, j + 2] if j + 2 < self.width else 0
					context[2] = obs[i, j + 1] if j + 1 < self.width else 0
					context[1] = obs[i, j - 1] if j - 1 > 0 else 0
					context[0] = obs[i, j - 2] if j - 2 > 0 else 0

				log_prob += cts_update(& self.cts_factors[i][j], context, obs[i, j])
				log_recoding_prob += cts_log_prob(& self.cts_factors[i][j], context, obs[i, j])
		#=======================================================================
		# log_prob /= self.height*self.width  # take averages to avoid value explosion
		# log_recoding_prob /= self.height*self.width  # take averages to avoid value explosion
		#=======================================================================

		return log_prob, log_recoding_prob

	def exploration_bonus(self, log_prob, log_recoding_prob):
		recoding_prob = np.exp(log_recoding_prob)
		prob_ratio = np.exp(log_recoding_prob - log_prob)

		pseudocount = (1 - recoding_prob) / np.maximum(prob_ratio - 1, 1e-10)
		return self.beta / np.sqrt(pseudocount + .01)

__all__ = ["CTS", "CTSDensityModel"]
