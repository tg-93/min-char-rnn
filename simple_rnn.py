# a simple, non-lstm rnn

import numpy as np
import os
from datetime import datetime

class VanillaRNN:
	def __init__(self, hidden_size, vocab_size, hprev = None):
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.h = hprev if hprev else np.zeros((hidden_size,1))
		# Model params
		self.Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
		self.Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
		self.Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
		self.bh = np.zeros((hidden_size, 1)) # hidden bias
		self.by = np.zeros((vocab_size, 1)) # output bias
		# memory variables for Adagrad
		self.mWxh, self.mWhh, self.mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
		self.mbh, self.mby = np.zeros_like(self.bh), np.zeros_like(self.by)

	def reset_memory(self):
		if self.h is not None:
			self.h = np.zeros_like(self.h)

	def save(self, name):
		current_date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
		name = f'vanilla_rnn_{name}_{current_date_time}'
		os.mkdir(name)
		np.save(name + '/Wxh.npy', self.Wxh)
		np.save(name + '/Whh.npy', self.Whh)
		np.save(name + '/Why.npy', self.Why)
		np.save(name + '/bh.npy', self.bh)
		np.save(name + '/by.npy', self.by)

	def sample(self, n, seed_ix):
		""" 
		sample a sequence of integers from the model 
		hprev is memory state, seed_ix is seed letter for first time step
		"""
		x = np.zeros((self.vocab_size, 1))
		x[seed_ix] = 1
		outputs = []
		h = self.h
		for t in range(n):
			h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
			y = np.dot(self.Why, h) + self.by
			probs = np.exp(y) / np.sum(np.exp(y))
			ix = np.random.choice(range(self.vocab_size), p=probs.ravel())
			x = np.zeros_like(x)
			x[ix] = 1
			outputs.append(ix)
		return outputs

	def training_step(self, inputs, targets, learning_rate):
		"""
		inputs,targets are both list of integers.
		returns the loss, gradients on model parameters, and last hidden state
		"""
		xs, hs, ys, probs = {}, {}, {}, {}
		hs[-1] = np.copy(self.h)
		loss = 0
		# forward pass
		for t in range(len(inputs)):
			xs[t] = np.zeros((self.vocab_size,1)) # encode in 1-of-k representation
			xs[t][inputs[t]] = 1
			hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state
			ys[t] = np.dot(self.Why, hs[t]) + self.by # unnormalized log probabilities for next chars
			probs[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
			loss += -np.log(probs[t][targets[t],0]) # softmax (cross-entropy loss)

		# backward pass: gradients
		dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
		dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
		dhnext = np.zeros_like(hs[0]) # derivative wrt h(t+1), since h(t) affects y(t+1) also.
		for t in reversed(range(len(inputs))):
			dy = np.copy(probs[t])
			dy[targets[t]] -= 1 # softmax derivative
			dWhy += np.dot(dy, hs[t].T)
			dby += dy
			dh = np.dot(self.Why.T, dy) + dhnext # backprop into h
			dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
			dbh += dhraw
			dWxh += np.dot(dhraw, xs[t].T)
			dWhh += np.dot(dhraw, hs[t-1].T)
			dhnext = np.dot(self.Whh.T, dhraw)

		for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
			np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

		self.h = hs[len(inputs)-1]
		# perform parameter update with Adagrad
		for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], 
			[dWxh, dWhh, dWhy, dbh, dby], 
			[self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):
			mem += dparam * dparam
			param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

		return loss