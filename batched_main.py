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
  tokenizer_filename = f'greedy_tokenizer_{args.vocab_size}_20240630.pkl'
  tokenizer = GreedyTokenizer.load(tokenizer_filename)
else:
  print("Starting tokenizer training.")
  tokenizer = GreedyTokenizer(data, args.vocab_size)
vocab_size = tokenizer.vocab_size()
print(f'vocab size after tokenizer training: {vocab_size}.')

encoded_data = tokenizer.encode(data)

data_size, observed_vocab_size = len(encoded_data), len(set(encoded_data))
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

smooth_loss = -np.log(1.0/vocab_size) # loss at iteration 0
iters_per_epoch = int(len(encoded_data)//(batch_size*seq_length))
epoch = 0
while epoch <= min(num_epochs, int(args.num_iter/iters_per_epoch)):
  epoch += 1
  print(f'=========== Epoch: {epoch} ============')
  model.reset_memory()
  print(f'hidden size: {hidden_size}. seq_length: {seq_length}')

  for n in range(iters_per_epoch):    
    input_end = data_size - seq_length - 1
    # randomly sample starting indices
    starts = np.random.randint(input_end, size=batch_size)
    # inputs and targets are of shape seq_length * batch_size
    inputs = [[encoded_data[i+j] for j in starts] for i in range(seq_length)]
    targets = [[encoded_data[i+j+1] for j in starts] for i in range(seq_length)]

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

  # Get a larger sample at the end of an epoch
  sample_ix = model.sample(400, inputs[-1][0])
  context = tokenizer.decode(inputs[:][0])
  print(f'*** Context: {context} ***')
  txt = tokenizer.decode(sample_ix)
  print('*** Sample: ***')
  print(f'----\n{txt}\n----')

  # checkpoint the model on fibonacci epochs
  if epoch in {1,2,3,5,8,13,21,34,55,89}:
      model.save(f'hsize{hidden_size}_seq{seq_length}_checkpoint@{epoch}')

model.save(f'hsize{hidden_size}_seq{seq_length}_final')