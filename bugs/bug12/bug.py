def compile_keras_sequential_model(list_of_layers, msg):

    # a tf.keras.Sequential model is a sequence of layers
    model = tf.keras.Sequential(list_of_layers)

    # keras does not have a pre-defined metric for Root Mean Square Error. Let's define one.
    def rmse(y_true, y_pred): # Root Mean Squared Error
      return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

    print('\nModel ', msg)

    #Optimizer
    sgd = tf.keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    # to finalize the model, specify the loss, the optimizer and metrics
    model.compile(
       loss = 'mean_squared_error',
       optimizer = sgd,
#         optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
       metrics = [rmse])

    # this prints a description of the model
    model.summary()

    return model
#Create Keras model
def model_fn_keras():

    # RNN model (RMSE: 0.164 after 10 epochs)
    model_layers_RNN = [
        l.Reshape([SEQLEN, 1], input_shape=[SEQLEN,]), # [BATCHSIZE, SEQLEN, 1] is necessary for RNN model
        l.GRU(RNN_CELLSIZE, return_sequences=True),  # output shape [BATCHSIZE, SEQLEN, RNN_CELLSIZE]
        l.GRU(RNN_CELLSIZE), # keep only last output in sequence: output shape [BATCHSIZE, RNN_CELLSIZE]
        l.Dense(1) # output shape [BATCHSIZE, 1]
    ]

    model_RNN = compile_keras_sequential_model(model_layers_RNN, "RNN")

    return(model_RNN)

#Convert
estimator = tf.keras.estimator.model_to_estimator(keras_model=model_fn_keras())