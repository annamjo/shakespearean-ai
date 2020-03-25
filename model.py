from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
# tf.keras.backend.clear_session() #use with 2.1.0-rc0 (very slow) as workaround?
#memory leak w/  pip install tensorflow==2.0.0

import numpy as np
import os
import time
tf.compat.v1.enable_eager_execution() #for 1.14.0, can run on Google colab

# download text data
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# read then decode
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# print num of characters in text = length
print('Length of text: {} characters'.format(len(text)))

print(text[:250]) # 1st 250 chars in text

# unique chars in file
vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))

# vectorize: mapping chars to numerical indices
char2idx =  {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

#each char = index from 0 to length of vocab: len(vocab) - unique chars
print('{') # printing array of eash unique char
for char,_ in zip(char2idx, range(20)):
    print('{:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')

# show how 1st 13 chars, 'first citizen', rep as ints (text->int) 
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

# model's task: find/predict the most probable next character given a character/sequence of characters
# recurrent neural networks (RNNs) depend on previously seen elements (all given chars from before)

seq_length = 100 #max length sentence (for a single input in chars)
examples_per_epoch = len(text)//(seq_length+1)

# training examples + targets: input example sequences of text
# each same length and 1 char shifted to right
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int) #tf.data to split text into seqs

for i in char_dataset.take(5):
    print(idx2char[i.numpy()])

#use batch method to convert individual chars into concatenated sequences
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))

#function used for each batch/sequence
# duplicate and shift it to form the input + target text
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text # target - shifted 1 char to right

dataset = sequences.map(split_input_target) # map method to apply func on all els of 'sequences'

for input_example, target_example in dataset.take(1):
    print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    print('Target data: ', repr(''.join(idx2char[target_example.numpy()])))

#for each time step, RNN model recieves an index for a char to predict the index for next char 
# RNN has to consider previous step & current input char
for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print('Step {:4d}'.format(i))
    print('  input: {} ({:s})'.format(input_idx, repr(idx2char[input_idx])))
    print('  expected output: {} ({:s})'.format(target_idx, repr(idx2char[target_idx])))

BATCH_SIZE = 64
#TF data maintains a buffer in which it shuffles elements
#TF data -> possibly infinite sequences, doesn't shuffle entire seq in memory
BUFFER_SIZE = 10000 #to shuffle dataset

# shuffle TF data and pack into training batches
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(dataset)

vocab_size = len(vocab) #length of the vocabulary in chars
embedding_dim = 256
rnn_units = 1024 #number of RNN units

#use Sequential to build/define the model - need 3 layers:
#embedding - input layer, trainable lookup table that will map a char's int/num to a vector w/embedding_dim dimensions
#GRU - type of RNN w/ size = rnn_units (could also use a LSTM layer)
#Dense - output layer, has vocab_size outputs
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

#input char (batch data) -> embedding layer -> char embedding -> GRU -> GRU output -> dense layer -> logits (probabs/predictions)
model = build_model( #GRU changes state: before->after
    vocab_size = len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)
#for each char: model looks embedding -> embedding is input for GRU, which runs one timestep
#model applies dense layer to generate logits predicting the log-likelihood of the next character!

#run the model, check behavior:
for input_example_batch, target_example_batch in dataset.take(1): #check shape of output
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, '# (batch_size, sequence_length, vocab_size)')
# here, input's sequence length is 10, but model can run on inputs of any length

model.summary()

#to get actual predictions from model, need to sample from output distribution - get actual char indices
#output dist = logits over char vocabulary; Must sample, not take argmax of dist (would get model stuck in loop)
#try for 1st example in batch:
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

print(sampled_indices) # gives prediction of next char index at each timestep

# decode char indices ((int back to char) to see text predicted by UNTRAINED model
print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices]))) #gibberish as of now

# TRAIN the model:
# given previous RNN state (prev time step) + the input this time step, predict the class of next char
def loss(labels, logits): #model returns logits, so set from_logits flag
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
#attached a loss function^, applied across last dimension of the predictions

example_batch_loss = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(optimizer='adam', loss=loss) #attached optimizer, to configure training procedure

#configure checkpts:
checkpoint_dir = './training_checkpoints' #directry where ckpts will be saved
checkpoint_prefix  = os.path.join(checkpoint_dir, "ckpt_{epoch}") #name of ckpt files

#to ensure chkpts are saved during training
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

#execute the training: use 10 epochs train model (for reasonable training time)
EPOCHS = 10
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

###generate text: 
tf.train.latest_checkpoint(checkpoint_dir)

#model accepts a fixed batch size once built
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir)) #weights from checkpoint
model.build(tf.TensorShape([1, None]))

model.summary()

###prediction loop:
#func chooses a start string (to initialize RNN state + set # of chars to generate)
#gets prediction dist for next char using start string + RNN state
#uses categorical dist to calc index of predicted char = next input to model
#after predicting next char, (modified) RNN state(s) returned by model...
# fed back into model, so now model gets more context, not just previoudly predicted chars
# def generate_text(model, start_string):
