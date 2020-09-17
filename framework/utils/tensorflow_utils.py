# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa
import numpy as np

def separate(i, value):
	true_labels = tf.ones_like(i)
	false_labels = tf.zeros_like(i)
	mask = tf.where(tf.greater_equal(i, value), true_labels, false_labels)
	greater_equal = mask*i
	lower = i - greater_equal
	return greater_equal, lower

def get_optimization_function(name):
	if hasattr(tf.keras.optimizers, name):
		return eval('tf.keras.optimizers.'+name)
	# if hasattr(tf.contrib.opt, opt_name):
	# 	return eval('tf.contrib.opt.'+opt_name)
	if hasattr(tfa.optimizers, name):
		return eval('tfa.optimizers.'+name)
	return None
	
def get_annealable_variable(function_name, initial_value, global_step, decay_steps, decay_rate):
	return eval('tf.train.'+function_name)(learning_rate=initial_value, global_step=global_step, decay_steps=decay_steps, decay_rate=decay_rate)
	
def get_available_gpus():
	# recipe from here: https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
	local_device_protos = tf.config.list_physical_devices('GPU')
	return [x.name for x in local_device_protos]

def gpu_count():
	return len(get_available_gpus())

def orthogonal_initializer(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init
