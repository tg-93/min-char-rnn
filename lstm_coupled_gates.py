# LSTM with no peepholes, coupled f & i gates and vectorised batch training

import numpy as np
import os
from datetime import datetime

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

class CoupledLSTM:
	def __init__(self, hidden_size, vocab_size, batch_size):
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.batch_size = batch_size
		self.c = np.zeros((hidden_size, batch_size))
		self.h = np.zeros((hidden_size, batch_size))
		## Model params
		# forget gate
		self.Wfioc = np.random.randn(3*hidden_size, vocab_size+hidden_size)*0.01
		self.bfioc = np.zeros((3*hidden_size, 1))
		# generate y from h
		self.Why = np.random.randn(vocab_size, hidden_size) * 0.01
		self.by = np.zeros((vocab_size, 1))
		# memory variables for Adagrad
		self.mWfioc, self.mbfioc = np.zeros_like(self.Wfioc), np.zeros_like(self.bfioc)
		self.mWhy, self.mby = np.zeros_like(self.Why), np.zeros_like(self.by)

	def sample(self, n, seed_ix):
		""" 
		sample a sequence of integers from the model 
		hprev is memory state, seed_ix is seed letter for first time step
		"""
		x = np.zeros((self.vocab_size, 1))
		x[seed_ix] = 1
		outputs = []
		h = self.h[:,0].reshape((self.hidden_size, 1))
		c = self.c[:,0].reshape((self.hidden_size, 1))
		for t in range(n):
			xh = np.vstack((x, h))
			fioc = np.dot(self.Wfioc, xh) + self.bfioc
			fio = sigmoid(fioc[:2*self.hidden_size])
			# generate new C
			c_new = np.tanh(fioc[2*self.hidden_size:])
			c = (fio[:self.hidden_size]*c) + ((1.0 - fio[:self.hidden_size])*c_new)
			# generate new h
			h = fio[self.hidden_size:2*self.hidden_size]*np.tanh(c)
			y = np.dot(self.Why, h) + self.by
			probs = np.exp(y) / np.sum(np.exp(y))
			ix = np.random.choice(range(self.vocab_size), p=probs.ravel())
			x = np.zeros_like(x)
			x[ix] = 1
			outputs.append(ix)
		return outputs

	def training_step(self, inputs, targets, learning_rate):
		"""
		inputs,targets are both matrices of size n*b, n is sequence length and b is batch size.
		returns the loss, gradients on model parameters, and last hidden state
		"""
		xs, cs, hs, h_new, c_new, fgate, ogate, probs = {}, {}, {}, {}, {}, {}, {}, {}
		cs[-1] = np.copy(self.c)
		hs[-1] = np.copy(self.h)
		loss = 0
		# forward pass
		for t in range(len(inputs)):
			# encode in 1-of-k representation
			xs[t] = np.zeros((self.vocab_size,self.batch_size))
			for i, ix in enumerate(inputs[t]):
				xs[t][ix, i] = 1
			xh = np.vstack((xs[t], hs[t-1]))
			#assert xh.shape==(self.vocab_size+self.hidden_size, self.batch_size)
			
			# calculate gates
			fioc = np.dot(self.Wfioc, xh) + self.bfioc #output (3*hidden_size, batch_size)
			#assert fioc.shape==(3*self.hidden_size, self.batch_size)
			
			fio = sigmoid(fioc[:2*self.hidden_size, :])
			fgate[t] = fio[:self.hidden_size, :]
			#assert fgate[t].shape==(self.hidden_size, self.batch_size)
			
			ogate[t] = fio[self.hidden_size:2*self.hidden_size,:]
			#assert ogate[t].shape==(self.hidden_size, self.batch_size)

			# generate new C
			c_new[t] = np.tanh(fioc[2*self.hidden_size:,:])
			#assert c_new[t].shape==(self.hidden_size, self.batch_size)

			cs[t] = (fgate[t]*cs[t-1]) + ((1.0 - fgate[t])*c_new[t])
			# generate new h
			h_new[t] = np.tanh(cs[t])
			hs[t] = ogate[t]*h_new[t]
			# map to y
			y = np.dot(self.Why, hs[t]) + self.by
			#assert y.shape == (self.vocab_size, self.batch_size)

			probs[t] = np.exp(y) / np.sum(np.exp(y), axis=0) # probabilities for next chars
			#assert probs[t].shape == (self.vocab_size, self.batch_size)
			
			for i, ix in enumerate(targets[t]):
				loss += -np.log(probs[t][ix, i]) # softmax (cross-entropy loss)
		loss /= self.batch_size

		# backward pass: gradients
		dWfioc, dbfioc = np.zeros_like(self.Wfioc), np.zeros_like(self.bfioc)
		dWhy, dby = np.zeros_like(self.Why), np.zeros_like(self.by)
		
		dhnext = np.zeros_like(self.h) # derivative wrt h(t+1), since h(t) is propagated to t+1.
		dcnext = np.zeros_like(self.c) # derivative wrt c(t+1), since c(t) is propagated to t+1.
		for t in reversed(range(len(inputs))):
			# backprop cross entropy loss through softmax
			dy = np.copy(probs[t])
			for i, ix in enumerate(targets[t]):
				dy[ix, i] -= 1.0
			dWhy += np.dot(dy, hs[t].T)
			dby += np.sum(dy)

			# backprop to h & add backprop from t+1
			dh = np.dot(self.Why.T, dy) + dhnext

			xh = np.vstack((xs[t], hs[t-1]))
			# backprop through o-gate (sigmoid)
			do_raw = dh*hs[t]*(1.0 - ogate[t])
			
			# backprop to c
			dc = dh*ogate[t]*(1.0 - h_new[t]*h_new[t]) + dcnext

			# backprop before i-gate and through tanh
			dc_raw = dc*(1.0 - fgate[t])*(1.0 - c_new[t]*c_new[t])

			# backprop to f-gate and i-gate params
			df_raw = dc*(cs[t-1] - c_new[t])*fgate[t]*(1.0 - fgate[t])

			dfioc_raw = np.vstack((df_raw,  do_raw, dc_raw))
			dWfioc += np.dot(dfioc_raw, xh.T)
			dbfioc += np.sum(dfioc_raw)
			
			# backprop to h(t-1) and c(t-1)
			dcnext = dc*fgate[t]
			dxh = np.dot(self.Wfioc.T, dfioc_raw)
			dhnext = dxh[self.vocab_size:]

		for dparam in [dWfioc, dbfioc, dWhy, dby]:
			np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

		self.h = hs[len(inputs)-1]
		self.c = cs[len(inputs)-1]
		# perform parameter update with Adagrad
		for param, dparam, mem in zip(
			[self.Wfioc, self.bfioc, self.Why, self.by], 
			[dWfioc, dbfioc, dWhy, dby], 
			[self.mWfioc, self.mbfioc, self.mWhy, self.mby]):
			mem += dparam * dparam
			param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

		return loss

	def reset_memory(self):
		if self.c is not None:
			self.c = np.zeros_like(self.c)
		if self.h is not None:
			self.h = np.zeros_like(self.h)

	def save(self, name):
		current_date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
		name = f'coupled_lstm_{name}_{current_date_time}'
		os.mkdir(name)
		np.save(name + '/Wfioc.npy', self.Wfioc)
		np.save(name + '/Why.npy', self.Why)
		np.save(name + '/bfioc.npy', self.bfioc)
		np.save(name + '/by.npy', self.by)