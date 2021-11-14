# https://stackoverflow.com/questions/62907044/why-model-get-weights-is-empty-is-tensorflow-bug
import tensorflow as tf
from tensorflow.keras import layers, activations, losses, Model, optimizers, models
import numpy as np


class MAMLmodel(Model):
    def __init__(self):
        super().__init__()

        self.Dense1 = layers.Dense(2, input_shape=(3, ))
        self.Dense2 = layers.Dense(1)

    def forward(self, inputs):
        x = self.Dense1(inputs)
        x = self.Dense2(x)

        return x

def compute_loss(y_true, y_pred):
    return losses.mean_squared_error(y_true, y_pred)

x1 = [[[1], [1], [1]],
      [[1], [1], [1]],
      [[1], [1], [1]]]

y1 = [[[0], [0], [0]],
      [[0], [0], [0]],
      [[0], [0], [0]]]
x1 = tf.convert_to_tensor(x1)
y1 = tf.convert_to_tensor(y1) 

inner_train_step = 1
batch_size = 3
lr_inner = 0.001

model = MAMLmodel()
inner_optimizer = optimizers.Adam()

for i in range(batch_size):
    # If inner_train_step is 2 or bigger, the gradient is empty list.
    for inner_step in range(inner_train_step):
        with tf.GradientTape() as support_tape:
            support_tape.watch(model.trainable_variables)
            y_pred = model.forward(x1[i])
            support_loss = compute_loss(y1[i], y_pred)

        gradients = support_tape.gradient(support_loss, model.trainable_variables)
        # inner_optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        k = 0
        for j in range(len(model.layers)):
            print(type(model.layers[j].kernel))
            model.layers[j].kernel = tf.subtract(model.layers[j].kernel, tf.multiply(lr_inner, gradients[k]))
            model.layers[j].bias = tf.subtract(model.layers[j].bias, tf.multiply(lr_inner, gradients[k + 1]))
            print(type(model.layers[j].kernel))
            # print(model.layers[j].bias)
            k += 2

    # If you use 'optimizer.apply_gradients' update gradient,it can print weights.
    # But if you update gradient by yourself,it just print empty list.
    # print(model.get_weights())