import tensorflow as tf
print(tf.__version__)
import datetime, os

fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

def train_model():

  model = create_model()
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  #NAME = "Trains_vs_Cars_16by2_CNN_{}".format(int(time.time()))
  NAME = "Trains_vs_Cars_16by2_{}".format(str(datetime.datetime.now()))
  tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(NAME))

model.fit(X, y,
      batch_size=25,
      epochs=5,
      validation_split=0.2,
      callbacks=[tensorboard])

train_model()
