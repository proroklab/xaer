# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.compat.v1 as tf

from .ops import causal_conv

# https://github.com/ibab/tensorflow-wavenet

def create_filter_variable(shape):
	'''Create a convolution filter variable with the specified name and shape,
	and initialize it using Xavier initialition.'''
	initializer = tf.contrib.layers.xavier_initializer_conv2d()
	return initializer(shape=shape)

def create_embedding_table(shape):
	if shape[0] == shape[1]:
		# Make a one-hot encoding as the initial value.
		return np.identity(n=shape[0], dtype=np.float32)
	else:
		return create_filter_variable(shape)


def create_bias_variable(shape):
	'''Create a bias variable with the specified name and shape and initialize
	it to zero.'''
	initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
	return initializer(shape=shape)

class WaveNetModel(object):
	'''Implements the WaveNet network for generative audio.

	Usage (with the architecture as in the DeepMind paper):
		dilations = [2**i for i in range(N)] * M
		filter_width = 2  # Convolutions just use 2 samples.
		residual_channels = 16  # Not specified in the paper.
		dilation_channels = 32  # Not specified in the paper.
		skip_channels = 16	  # Not specified in the paper.
		net = WaveNetModel(dilations, filter_width,
						   residual_channels, dilation_channels,
						   skip_channels)
		loss = net.loss(input_batch)
	'''

	def __init__(self,
				 dilations,
				 filter_width,
				 residual_channels,
				 dilation_channels,
				 skip_channels,
				 quantization_channels=2**8,
				 use_biases=False,
				 scalar_input=False,
				 initial_filter_width=32,
				 histograms=False,
				 global_condition_channels=None,
				 global_condition_cardinality=None):
		'''Initializes the WaveNet model.

		Args:
			dilations: A list with the dilation factor for each layer.
			filter_width: The samples that are included in each convolution,
				after dilating.
			residual_channels: How many filters to learn for the residual.
			dilation_channels: How many filters to learn for the dilated
				convolution.
			skip_channels: How many filters to learn that contribute to the
				quantized softmax output.
			quantization_channels: How many amplitude values to use for audio
				quantization and the corresponding one-hot encoding.
				Default: 256 (8-bit quantization).
			use_biases: Whether to add a bias layer to each convolution.
				Default: False.
			scalar_input: Whether to use the quantized waveform directly as
				input to the network instead of one-hot encoding it.
				Default: False.
			initial_filter_width: The width of the initial filter of the
				convolution applied to the scalar input. This is only relevant
				if scalar_input=True.
			histograms: Whether to store histograms in the summary.
				Default: False.
			global_condition_channels: Number of channels in (embedding
				size) of global conditioning vector. None indicates there is
				no global conditioning.
			global_condition_cardinality: Number of mutually exclusive
				categories to be embedded in global condition embedding. If
				not None, then this implies that global_condition tensor
				specifies an integer selecting which of the N global condition
				categories, where N = global_condition_cardinality. If None,
				then the global_condition tensor is regarded as a vector which
				must have dimension global_condition_channels.

		'''
		self.dilations = dilations
		self.filter_width = filter_width
		self.residual_channels = residual_channels
		self.dilation_channels = dilation_channels
		self.quantization_channels = quantization_channels
		self.use_biases = use_biases
		self.skip_channels = skip_channels
		self.scalar_input = scalar_input
		self.initial_filter_width = initial_filter_width
		self.histograms = histograms
		self.global_condition_channels = global_condition_channels
		self.global_condition_cardinality = global_condition_cardinality

		self.receptive_field = WaveNetModel.calculate_receptive_field(
			self.filter_width, self.dilations, self.scalar_input,
			self.initial_filter_width)
		self.variables = self._create_variables()

	@staticmethod
	def calculate_receptive_field(filter_width, dilations, scalar_input, initial_filter_width):
		receptive_field = (filter_width - 1) * sum(dilations) + 1
		if scalar_input:
			receptive_field += initial_filter_width - 1
		else:
			receptive_field += filter_width - 1
		return receptive_field

	def _create_variables(self):
		'''This function creates all variables used by the network.
		This allows us to share them between multiple calls to the loss
		function and generation function.'''

		var = dict()

		if self.global_condition_cardinality is not None:
			# We only look up the embedding if we are conditioning on a
			# set of mutually-exclusive categories. We can also condition
			# on an already-embedded dense vector, in which case it's
			# given to us and we don't need to do the embedding lookup.
			# Still another alternative is no global condition at all, in
			# which case we also don't do a tf.nn.embedding_lookup.
			layer = dict()
			layer['gc_embedding'] = create_embedding_table([self.global_condition_cardinality, self.global_condition_channels])
			var['embeddings'] = layer

		layer = dict()
		if self.scalar_input:
			initial_channels = 1
			initial_filter_width = self.initial_filter_width
		else:
			initial_channels = self.quantization_channels
			initial_filter_width = self.filter_width
		layer['filter'] = create_filter_variable([initial_filter_width, initial_channels, self.residual_channels])
		var['causal_layer'] = layer
		var['dilated_stack'] = list()
		for i, dilation in enumerate(self.dilations):
			current = dict()
			current['filter'] = create_filter_variable([self.filter_width, self.residual_channels, self.dilation_channels])
			current['gate'] = create_filter_variable([self.filter_width, self.residual_channels, self.dilation_channels])
			current['dense'] = create_filter_variable([1, self.dilation_channels, self.residual_channels])
			current['skip'] = create_filter_variable([1, self.dilation_channels, self.skip_channels])

			if self.global_condition_channels is not None:
				current['gc_gateweights'] = create_filter_variable([1, self.global_condition_channels, self.dilation_channels])
				current['gc_filtweights'] = create_filter_variable([1, self.global_condition_channels, self.dilation_channels])

			if self.use_biases:
				current['filter_bias'] = create_bias_variable([self.dilation_channels])
				current['gate_bias'] = create_bias_variable([self.dilation_channels])
				current['dense_bias'] = create_bias_variable([self.residual_channels])
				current['skip_bias'] = create_bias_variable([self.skip_channels])

			var['dilated_stack'].append(current)
		return var

	def _create_causal_layer(self, input_batch):
		'''Creates a single causal convolution layer.

		The layer can change the number of channels.
		'''
		weights_filter = self.variables['causal_layer']['filter']
		return causal_conv(input_batch, weights_filter, 1)

	def _create_dilation_layer(self, input_batch, layer_index, dilation, global_condition_batch, output_width):
		'''Creates a single causal dilated convolution layer.

		Args:
			 input_batch: Input to the dilation layer.
			 layer_index: Integer indicating which layer this is.
			 dilation: Integer specifying the dilation size.
			 global_conditioning_batch: Tensor containing the global data upon
				 which the output is to be conditioned upon. Shape:
				 [batch size, 1, channels]. The 1 is for the axis
				 corresponding to time so that the result is broadcast to
				 all time steps.

		The layer contains a gated filter that connects to dense output
		and to a skip connection:

			   |-> [gate]   -|		|-> 1x1 conv -> skip output
			   |			 |-> (*) -|
		input -|-> [filter] -|		|-> 1x1 conv -|
			   |									|-> (+) -> dense output
			   |------------------------------------|

		Where `[gate]` and `[filter]` are causal convolutions with a
		non-linear activation at the output. Biases and global conditioning
		are omitted due to the limits of ASCII art.

		'''
		variables = self.variables['dilated_stack'][layer_index]

		weights_filter = variables['filter']
		weights_gate = variables['gate']

		conv_filter = causal_conv(input_batch, weights_filter, dilation)
		conv_gate = causal_conv(input_batch, weights_gate, dilation)

		if global_condition_batch is not None:
			weights_gc_filter = variables['gc_filtweights']
			conv_filter = conv_filter + tf.nn.conv1d(global_condition_batch, weights_gc_filter, stride=1, padding="SAME")
			weights_gc_gate = variables['gc_gateweights']
			conv_gate = conv_gate + tf.nn.conv1d(global_condition_batch, weights_gc_gate, stride=1, padding="SAME")

		if self.use_biases:
			filter_bias = variables['filter_bias']
			gate_bias = variables['gate_bias']
			conv_filter = tf.add(conv_filter, filter_bias)
			conv_gate = tf.add(conv_gate, gate_bias)

		out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

		# The 1x1 conv to produce the residual output
		weights_dense = variables['dense']
		transformed = tf.nn.conv1d(out, weights_dense, stride=1, padding="SAME")

		# The 1x1 conv to produce the skip output
		skip_cut = out.get_shape().as_list()[1] - output_width
		out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, -1])
		weights_skip = variables['skip']
		skip_contribution = tf.nn.conv1d(out_skip, weights_skip, stride=1, padding="SAME")

		if self.use_biases:
			dense_bias = variables['dense_bias']
			skip_bias = variables['skip_bias']
			transformed = transformed + dense_bias
			skip_contribution = skip_contribution + skip_bias

		#=======================================================================
		# if self.histograms:
		# 	layer = 'layer{}'.format(layer_index)
		# 	tf.histogram_summary(layer + '_filter', weights_filter)
		# 	tf.histogram_summary(layer + '_gate', weights_gate)
		# 	tf.histogram_summary(layer + '_dense', weights_dense)
		# 	tf.histogram_summary(layer + '_skip', weights_skip)
		# 	if self.use_biases:
		# 		tf.histogram_summary(layer + '_biases_filter', filter_bias)
		# 		tf.histogram_summary(layer + '_biases_gate', gate_bias)
		# 		tf.histogram_summary(layer + '_biases_dense', dense_bias)
		# 		tf.histogram_summary(layer + '_biases_skip', skip_bias)
		#=======================================================================

		input_cut = input_batch.get_shape().as_list()[1] - transformed.get_shape().as_list()[1]
		input_batch = tf.slice(input_batch, [0, input_cut, 0], [-1, -1, -1])

		return skip_contribution, input_batch + transformed

	def create_network(self, input_batch, global_condition_batch=None):
		'''Construct the WaveNet network.'''
		outputs = []
		current_layer = input_batch
		# Pre-process the input with a regular convolution
		current_layer = self._create_causal_layer(current_layer)
		output_width = input_batch.get_shape().as_list()[1] - self.receptive_field + 1
		# Add all defined dilation layers.
		for layer_index, dilation in enumerate(self.dilations):
			output, current_layer = self._create_dilation_layer(current_layer, layer_index, dilation, global_condition_batch, output_width)
			outputs.append(output)
		total = sum(outputs)
		return total
