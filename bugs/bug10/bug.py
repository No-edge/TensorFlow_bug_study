import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import os
import pickle
import cv2
from sklearn import model_selection as ms
from nets import inception_v1,inception_utils
import math

def one_hot_matrix(labels, C):

    C = tf.constant(C,name='C')
    one_hot_matrix = tf.one_hot(labels,C,axis=0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()

    return one_hot


def make_mini_batches(X, Y, mini_batch_size):

    m = X.shape[0]                  
    mini_batches = []

    # number of mini batches of size mini_batch_size in the dataset
    num_complete_minibatches = math.floor(m/mini_batch_size) 

    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[k*mini_batch_size : (k+1)*mini_batch_size,...]
        mini_batch_Y = Y[k*mini_batch_size : (k+1)*mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = X[num_complete_minibatches*mini_batch_size:,...]
        mini_batch_Y = Y[num_complete_minibatches*mini_batch_size:,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

# function to read the batches
def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    datadict = {'data':features,'labels':labels}

    return datadict

# combine batches into one dataset (batch size: 10000)
full_data = load_cfar10_batch('./cifar_10',1)['data']
full_labels = []
for i in range(5):
    full_labels.extend(load_cfar10_batch('./cifar_10',i+1)['labels'])
    if i > 0:
        full_data = np.concatenate((full_data,load_cfar10_batch('./cifar_10',i+1)['data']),axis = 0)

# dataset sizes
full_data.shape, len(full_labels)

# data preprocessing (using only 1/10 of the dataset for speed)

X = full_data[0:5000]           
y = one_hot_matrix(full_labels[0:5000], 10).T       

# split into training-validation sets
x_train, x_val, y_train, y_val = ms.train_test_split(X, y, test_size=0.2, random_state=1)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')

x_train = x_train / 255.0
x_val = x_val / 255.0

print('x_train shape:',x_train.shape)
print('y_train shape:',y_train.shape)
print('x_val shape:',x_val.shape)
print('y_val shape:',y_val.shape)

with tf.Session() as sess:

    sess.run(init)

    for epoch in range(num_epochs):

        # learning rate decay
        if epoch % 8 == 0:
            learning_rate *= math.pow(0.95,epoch/8)

        minibatch_cost = 0.

        for minibatch in minibatches:

            (minibatch_X, minibatch_Y) = minibatch
            _ , temp_cost = sess.run([optimizer, cost], feed_dict={inputs: minibatch_X, labels: minibatch_Y})
            minibatch_cost += temp_cost / num_minibatches

        # Print the cost every epoch
        if print_cost == True and epoch % 5 == 0:
            print ("Cost after epoch %i: %f" % (epoch, minibatch_cost),", Learning rate: %f" %(learning_rate))
        if print_cost == True and epoch % 1 == 0:
            costs.append(minibatch_cost)

    # Plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('Cost')
    plt.xlabel('Iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # Calculate the correct predictions
    predict_op = tf.argmax(Z, 1)
    correct_prediction = tf.equal(predict_op, tf.argmax(labels, 1))

    # Calculate accuracy on the validation set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #print(accuracy)
    train_accuracy = accuracy.eval({inputs: x_train, labels: y_train})
    val_accuracy = accuracy.eval({inputs: x_val, labels: y_val})
    print("Train Accuracy:", train_accuracy)
    print("Validation Accuracy:", val_accuracy)