# input processing, training, and sampling.
# Requires a sampling procedure and loss function defined separately.
import numpy as np
import math
from greedy_tokenizer import GreedyTokenizer
from lstm_batched import BatchedLSTM
from lstm_coupled_gates import CoupledLSTM
from lstm_decoupling import DecouplingLSTM
from stacked_lstm import StackedLSTM
import argparse

# Create the parser
parser = argparse.ArgumentParser()
# Add arguments
parser.add_argument("-epochs", "--num_epochs", help="Number of epochs", type=int,default=30)
parser.add_argument("-n", "--num_iter", help="Number of total iterations", type=int,default=10000000)
parser.add_argument("-hidden", "--hidden_size", help="size of hidden layer", type=int,default=256)
parser.add_argument("-seq", "--sequence_length", help="length of training sequence", type=int,default=64)
parser.add_argument("-batch", "--batch_size", help="number of sequences to train from at a time", type=int,default=16)
parser.add_argument("-m", "--model", help="type of model: lstm, coupled_lstm, decoupling_lstm, stacked_lstm", type=str, default="")
parser.add_argument("-l", "--layers", help="number of hidden layers", type=int, default=2)
parser.add_argument("-voc", "--vocab_size", help="size of tokenized vocab", type=int, default=200)
parser.add_argument("-rt", "--retrain_tokenizer", help="retrain tokenizer", type=bool, default=False)

# Parse the arguments
args = parser.parse_args()

# data I/O
data = open('wot1.txt', 'r').read() # should be simple plain text file
data += open('wot2.txt', 'r').read()
data += open('wot3.txt', 'r').read()
chars = sorted(list(set(data)))
data_size, vocab_size = len(data), len(chars)
print(f'data has {data_size} characters, {vocab_size} unique.')
if args.vocab_size in [100, 150, 200, 250, 400, 500, 600, 800, 1000] and not args.retrain_tokenizer:
  tokenizer_filename = f'greedy_tokenizer_{args.vocab_size}_20240628.pkl'
  tokenizer = GreedyTokenizer.load(tokenizer_filename)
else:
  print("Starting tokenizer training.")
  tokenizer = GreedyTokenizer(data, args.vocab_size)
vocab_size = tokenizer.vocab_size()
print(f'vocab size after tokenizer training: {vocab_size}.')

encoded_data = tokenizer.encode(data)

data_size, observed_vocab_size = len(data), len(chars)
print(f'Encoded data has {data_size} tokens, {observed_vocab_size} unique.')

# common hyperparameters
hidden_size = args.hidden_size # size of hidden layer of neurons
seq_length = args.sequence_length # number of steps to unroll the RNN for
num_epochs = args.num_epochs
batch_size = args.batch_size
learning_rate = 1e-1

if args.model == "lstm":
  model = BatchedLSTM(hidden_size, vocab_size, batch_size)
elif args.model == "coupled_lstm":
  model = CoupledLSTM(hidden_size, vocab_size, batch_size)
  print("Running Coupled gates model")
elif args.model == "decoupling_lstm":
  model = DecouplingLSTM(hidden_size, vocab_size, batch_size)
  print("Running decoupling model")
elif args.model == "stacked_lstm":
  num_layers = args.layers
  model = StackedLSTM(hidden_size, vocab_size, num_layers)
else:
  raise Exception("Invalid Model name")

p = 0 # data pointer 

smooth_loss = -np.log(1.0/vocab_size) # loss at iteration 0

batch_gap = int(len(encoded_data)//batch_size)

epoch = 0
n = 0
while epoch <= num_epochs and n < args.num_iter:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  # batch_gap is effectively the start of 2nd batch, so 1st batch should end before that
  if p+seq_length+1 >= batch_gap or n == 0:
    if n != 0:
      # Get a larger sample at the end of an epoch
      sample_ix = model.sample(400, inputs[-1][0])
      context = tokenizer.decode(inputs[:][0])
      print(f'*** Context: {context} ***')
      txt = tokenizer.decode(sample_ix)
      print('*** Sample: ***')
      print(f'----\n{txt}\n----')
    model.reset_memory()
    p = 0 # go from start of data
    epoch += 1
    print(f'=========== Epoch: {epoch} ============')
    print(f'hidden size: {hidden_size}. seq_length: {seq_length}')
    # checkpoint the model on fibonacci epochs
    if n>0 and epoch in {1,2,3,5,8,13,21,34,55,89}:
      model.save(f'hsize{hidden_size}_seq{seq_length}_ep{num_epochs}_checkpoint@{epoch}')
  
  input_end = batch_gap*batch_size
  # inputs and targets are of shape seq_length * batch_size
  inputs = [encoded_data[p+i : input_end: batch_gap] for i in range(seq_length)]
  targets = [encoded_data[p+i+1 : input_end + 1 : batch_gap] for i in range(seq_length)]

  # sample from the model now and then
  if n>0 and n%100 == 0:
    sample_ix = model.sample(200, inputs[0][0])
    txt = tokenizer.decode(sample_ix)
    print('*** Sample: ***')
    print(f'----\n{txt}\n----')

  # forward seq_length characters through the net and fetch gradient
  loss = model.training_step(inputs, targets, learning_rate)
  smooth_loss = smooth_loss * 0.995 + loss * 0.005
  if n>0 and n % 10 == 0:
    print(f'epoch: {epoch}, iter: {n}, loss: {smooth_loss}') # print progress

  p += seq_length # move data pointer
  n += 1
if epoch >= num_epochs:
  model.save(f'hsize{hidden_size}_seq{seq_length}_ep{num_epochs}')