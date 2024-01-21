# input processing, training, and sampling.
# Requires a sampling procedure and loss function defined separately.

import numpy as np
import math
from lstm_faster import LSTM
from simple_rnn import VanillaRNN
from hippo_rnn import HippoRNN
from hippo_lstm import HippoLSTM
import argparse

# Create the parser
parser = argparse.ArgumentParser()
# Add arguments
parser.add_argument("-epochs", "--num_epochs", help="Number of epochs", type=int,default=30)
parser.add_argument("-n", "--num_iter", help="Number of total iterations", type=int,default=10000000)
parser.add_argument("-hidden", "--hidden_size", help="size of hidden layer", type=int,default=256)
parser.add_argument("-seq", "--sequence_length", help="length of training sequence", type=int,default=64)
parser.add_argument("-m", "--model", help="type of model: vanilla_rnn, hippo_rnn, lstm", type=str, default="")

# Parse the arguments
args = parser.parse_args()

# data I/O
data = open('wot1.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print(f'data has {data_size} characters, {vocab_size} unique.')
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# common hyperparameters
hidden_size = args.hidden_size # size of hidden layer of neurons
seq_length = args.sequence_length # number of steps to unroll the RNN for
num_epochs = args.num_epochs
learning_rate = 1e-1

if args.model == "vanilla_rnn":
  model = VanillaRNN(hidden_size, vocab_size)
elif args.model == "lstm":
  model = LSTM(hidden_size, vocab_size)
elif args.model == "hippo_rnn":
  model = HippoRNN(hidden_size, min(seq_length//2, hidden_size//4), vocab_size, peephole=True)
elif args.model == "hippo_lstm":
  model = HippoLSTM(hidden_size, vocab_size, int(math.sqrt(hidden_size)))
else:
  raise Exception("Invalid Model name")

p = 0 # data pointer 

smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

epoch = 0
n = 0
while epoch <= num_epochs and n < args.num_iter:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0:
    if n != 0:
      # Get a larger sample at the end of an epoch
      sample_ix = model.sample(400, inputs[-1])
      context = ''.join(ix_to_char[ix] for ix in inputs)
      print(f'*** Context: {context} ***')
      txt = ''.join(ix_to_char[ix] for ix in sample_ix)
      print('*** Sample: ***')
      print(f'----\n{txt}\n----')
      epoch += 1
    model.reset_memory()
    p = 0 # go from start of data
    print(f'=========== Epoch: {epoch} ============')
    print(f'hidden size: {hidden_size}. seq_length: {seq_length}')
    # checkpoint the model on fibonacci epochs
    if epoch in {1,2,3,5,8,13,21,34,55,89}:
      model.save(f'hsize{hidden_size}_seq{seq_length}_ep{num_epochs}_checkpoint@{epoch}')
    
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n>0 and n % (100*(epoch+1)) == 0:
    sample_ix = model.sample(200, inputs[0])
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('*** Sample: ***')
    print(f'----\n{txt}\n----')

  # forward seq_length characters through the net and fetch gradient
  loss = model.training_step(inputs, targets, learning_rate)
  smooth_loss = smooth_loss * 0.995 + loss * 0.005
  if n>0 and n % (10*(epoch+1)) == 0:
    print(f'epoch: {epoch}, iter: {n}, loss: {smooth_loss}') # print progress

  p += seq_length # move data pointer
  n += 1
if epoch >= num_epochs:
  model.save(f'hsize{hidden_size}_seq{seq_length}_ep{num_epochs}')