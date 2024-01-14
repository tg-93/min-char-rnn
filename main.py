# input processing, training, and sampling.
# Requires a sampling procedure and loss function defined separately.

import numpy as np
import math
from lstm_faster import LSTM

# data I/O
data = open('wot1.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print(f'data has {data_size} characters, {vocab_size} unique.')
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# common hyperparameters
hidden_size = 256 # size of hidden layer of neurons
seq_length = 64 # number of steps to unroll the RNN for
learning_rate = 1e-1

#model = VanillaRNN(hidden_size, vocab_size)
model = LSTM(hidden_size, vocab_size)

p = 0 # data pointer 

smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

num_epoch = 30
epoch = 0
n = 0
while epoch <= num_epoch:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0:
    if n != 0:
      sample_ix = model.sample(400, inputs[-1])
      txt = ''.join(ix_to_char[ix] for ix in sample_ix)
      print(f'----\n{txt}\n----')
    model.reset_memory()
    p = 0 # go from start of data
    epoch += 1
    print(f'=========== Epoch: {epoch} ============')
    print(f'hidden size: {hidden_size}. seq_length: {seq_length}')
    if epoch % 5 == 0:
      model.save(f'lstm_{hidden_size}_{seq_length}_{num_epoch}_checkpoint@{epoch}_')
    
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n>0 and n % 5*max(int(math.log2(n)), 10) == 0:
    sample_ix = model.sample(200, inputs[0])
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print(f'----\n{txt}\n----')

  # forward seq_length characters through the net and fetch gradient
  loss = model.training_step(inputs, targets, learning_rate)
  smooth_loss = smooth_loss * 0.995 + loss * 0.005
  if n>0 and n % max(int(math.log2(n)), 10) == 0:
    print(f'epoch: {epoch}, iter: {n}, loss: {smooth_loss}') # print progress

  p += seq_length # move data pointer
  n += 1
model.save(f'lstm_{hidden_size}_{seq_length}_{num_epoch}_')