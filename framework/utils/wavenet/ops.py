# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf

# https://github.com/ibab/tensorflow-wavenet

def time_to_batch(value, dilation):
	shape = tf.shape(value)
	pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
	padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
	reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
	transposed = tf.transpose(reshaped, perm=[1, 0, 2])
	return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])

def batch_to_time(value, dilation):
	shape = tf.shape(value)
	prepared = tf.reshape(value, [dilation, -1, shape[2]])
	transposed = tf.transpose(prepared, perm=[1, 0, 2])
	return tf.reshape(transposed, [tf.div(shape[0], dilation), -1, shape[2]])

def causal_conv(value, filter_, dilation):
	filter_width = filter_.get_shape().as_list()[0]
	if dilation > 1:
		transformed = time_to_batch(value, dilation)
		conv = tf.nn.conv1d(transformed, filter_, stride=1, padding='VALID')
		restored = batch_to_time(conv, dilation)
	else:
		restored = tf.nn.conv1d(value, filter_, stride=1, padding='VALID')
	# Remove excess elements at the end.
	out_width = value.get_shape().as_list()[1] - (filter_width - 1) * dilation
	result = tf.slice(restored, [0, 0, 0], [-1, out_width, -1])
	return result