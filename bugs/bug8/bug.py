import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)
        self.build(input_shape=[None, 1])

    def call(self, inputs, **kwargs):
        return self.dense(inputs)

MyModel().summary()
model = MyModel()
tf.keras.utils.plot_model(model, to_file='bug.png', show_shapes=True)