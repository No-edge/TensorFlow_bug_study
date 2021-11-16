def extract_and_duplicate(tensor, reps=1, batch_size=0, sample_size=0):
    tensor = K.reshape(tensor[:,:,0],(batch_size, sample_size, 1))
    if reps > 1:
        tensor = Concatenate()([tensor for i in range(reps)])
    return tensor

input = Input(batch_shape = (batch_size, sample_size, num_features))
out = <steps to create a NN with several layers>

pre_mask = Lambda(extract_and_duplicate, arguments = {'reps': some_number,'batch_size': batch_size, 'sample_size': sample_size})(input)
mask = TimeDistributed(Dense(m, activation = 'tanh'))(out)
out = Multiply()([pre_mask,mask])
model = Model(input, out)
load_model(model_path, custom_objects={'estimated_accuracy': estimated_accuracy, 'extract_and_duplicate': extract_and_duplicate})
