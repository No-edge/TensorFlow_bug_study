import tensorflow as tf
s = tf.keras.layers.Input(shape=[2], dtype=tf.float32, name='s')
print(s._keras_shape)