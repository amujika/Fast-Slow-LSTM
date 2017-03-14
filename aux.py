#This file contains auxiliary functions

import numpy as np
import tensorflow as tf

#TODO: Change to use flags

def data_type():
  return tf.float16 if False else tf.float32

#Returns an orthogonal matrix of the given shape
def orthogonal(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)

#Returns an initializer that outputs an orthogonal matrix
def orthogonal_initializer(scale=1.0):
    def _initializer(shape, dtype=data_type(), partition_info=None):
        return tf.constant(orthogonal(shape) * scale, dtype)
    return _initializer


def layer_norm_all(h, base, num_units, scope):
    # Layer Norm (faster version)
    #
    # Performas layer norm on multiple base at once (ie, i, g, j, o for lstm)
    #
    # Reshapes h in to perform layer norm in parallel
    with tf.variable_scope(scope):
        h_reshape = tf.reshape(h, [-1, base, num_units])
        mean = tf.reduce_mean(h_reshape, [2], keep_dims=True)
        var = tf.reduce_mean(tf.square(h_reshape - mean), [2], keep_dims=True)
        epsilon = tf.constant(1e-3)
        rstd = tf.rsqrt(var + epsilon)
        h_reshape = (h_reshape - mean) * rstd
        # reshape back to original
        h = tf.reshape(h_reshape, [-1, base * num_units])

        alpha = tf.get_variable('layer_norm_alpha', [4 * num_units],
                                initializer=tf.constant_initializer(1.0), dtype=data_type())
        bias = tf.get_variable('layer_norm_bias', [4 * num_units],
                               initializer=tf.constant_initializer(0.0), dtype=data_type())

    return (h * alpha) + bias


def moments_for_layer_norm(x, axes=1, name=None):
    # output for mean and variance should be [batch_size]
    # from https://github.com/LeavesBreathe/tensorflow_with_latest_papers
    epsilon = 1e-3  # found this works best.
    if not isinstance(axes, list): axes = [axes]
    mean = tf.reduce_mean(x, axes, keep_dims=True)
    variance = tf.sqrt(tf.reduce_mean(tf.square(x - mean), axes, keep_dims=True) + epsilon)
    return mean, variance


def layer_norm(x, scope="layer_norm", alpha_start=1.0, bias_start=0.0):
    # derived from:
    # https://github.com/LeavesBreathe/tensorflow_with_latest_papers, but simplified.
    with tf.variable_scope(scope):
        num_units = x.get_shape().as_list()[1]

        alpha = tf.get_variable('alpha', [num_units],
                                initializer=tf.constant_initializer(alpha_start), dtype=data_type())
        bias = tf.get_variable('bias', [num_units],
                               initializer=tf.constant_initializer(bias_start), dtype=data_type())

        mean, variance = moments_for_layer_norm(x)
        y = (alpha * (x - mean)) / (variance) + bias
    return y

def zoneout(new_h, new_c, h, c, h_keep, c_keep, is_training):
  mask_c = tf.ones_like(c)
  mask_h = tf.ones_like(h)

  if is_training:
    mask_c = tf.nn.dropout(mask_c, c_keep)
    mask_h = tf.nn.dropout(mask_h, h_keep)

  mask_c *= c_keep
  mask_h *= h_keep

  h = new_h * mask_h + (-mask_h + 1.) * h
  c = new_c * mask_c + (-mask_c + 1.) * c

  return h, c
