import tensorflow as tf

a = tf.constant([2, 2, 3, 3], shape=[2, 2], dtype=tf.float32)
print('-------------------')
print(a)

a2 = tf.matmul(a,a)
print('-------------------')
print(a2)