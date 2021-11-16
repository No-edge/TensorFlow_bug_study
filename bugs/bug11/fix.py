import tensorflow as tf
from keras.layers import Input

s = Input(shape=[2], dtype=tf.float32, name='2')
s._shape_val # None
s._keras_shape # (None, 2)