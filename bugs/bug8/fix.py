import tensorflow as tf 

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)
        self.build(input_shape=[None, 1])

    def call(self, inputs, **kwargs):
        return self.dense(inputs)

    def build_graph(self):
        x = tf.keras.layers.Input(shape=(1))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

MyModel().build_graph().summary()
tf.keras.utils.plot_model(MyModel().build_graph(), to_file='fix.png', show_shapes=True)