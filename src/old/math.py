import tensorflow as tf


def to_float(x):
    return tf.cast(x, dtype=tf.float32)


def finite_reduce_sum(x):
    return tf.reduce_sum(x[tf.math.is_finite(x)])


