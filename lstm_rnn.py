# LSTM with no peepholes, and coupled input and forget gates

import numpy as np
import os
from datetime import datetime

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

class LSTM:
	def __init__(self, hidden_size, vocab_size):
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.c = np.zeros((hidden_size, 1))
		self.h = np.zeros((hidden_size, 1))
		## Model params
		# forget gate
		self.Wf = np.random.randn(hidden_size, vocab_size+hidden_size)*0.01
		self.bf = np.zeros((hidden_size, 1))
		# input gate
		self.Wi = np.random.randn(hidden_size, vocab_size+hidden_size)*0.01
		self.bi = np.zeros((hidden_size, 1))
		# output gate
		self.Wo = np.random.randn(hidden_size, vocab_size+hidden_size)*0.01
		self.bo = np.zeros((hidden_size, 1))
		# C update
		self.Wxc = np.random.randn(hidden_size, vocab_size+hidden_size)*0.01
		self.bc = np.zeros((hidden_size, 1))
		# generate y from h
		self.Why = np.random.randn(vocab_size, hidden_size) * 0.01
		self.by = np.zeros((vocab_size, 1))
		# memory variables for Adagrad
		self.mWf, self.mbf = np.zeros_like(self.Wf), np.zeros_like(self.bf)
		self.mWi, self.mbi = np.zeros_like(self.Wi), np.zeros_like(self.bi)
		self.mWo, self.mbo = np.zeros_like(self.Wo), np.zeros_like(self.bo)
		self.mWxc, self.mbc = np.zeros_like(self.Wxc), np.zeros_like(self.bc)
		self.mWhy, self.mby = np.zeros_like(self.Why), np.zeros_like(self.by)

	def reset_memory(self):
		if self.c is not None:
			self.c = np.zeros_like(self.c)
		if self.h is not None:
			self.h = np.zeros_like(self.h)

	def save(self, name):
		current_date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
		name = name + '_' + current_date_time
		os.mkdir(name)
		np.save(name + '/Wf.npy', self.Wf)
		np.save(name + '/Wi.npy', self.Wf)
		np.save(name + '/Wxc.npy', self.Wxc)
		np.save(name + '/Wo.npy', self.Wo)
		np.save(name + '/Why.npy', self.Why)
		np.save(name + '/bf.npy', self.bf)
		np.save(name + '/bi.npy', self.bf)
		np.save(name + '/bo.npy', self.bo)
		np.save(name + '/bc.npy', self.bc)
		np.save(name + '/by.npy', self.by)

	# def forward(self, input_ix, target):
	# 	x = np.zeros((self.vocab_size, 1))
	# 	x[input_ix] = 1
	# 	fgate = sigmoid(np.dot(self.Wf, x) + self.bf)
	# 	h_new = np.tanh(np.dot(self.Wxh, x) + self.bh)
	# 	self.h = (fgate*self.h) + (1.0-fgate)*h_new
	# 	ogate = sigmoid(np.dot(self.Wo, x) + self.bo)
	# 	y_a = np.dot(self.Why, self.h) + self.by
	# 	y = ogate*(np.tanh(y_a))
	# 	probs = np.exp(y) / np.sum(np.exp(y))
	# 	loss = -np.log(probs[target]) # softmax (cross-entropy loss)
	# 	return loss, probs, np.random.choice(range(self.vocab_size), p=probs.ravel())

	def sample(self, n, seed_ix):
		""" 
		sample a sequence of integers from the model 
		hprev is memory state, seed_ix is seed letter for first time step
		"""
		x = np.zeros((self.vocab_size, 1))
		x[seed_ix] = 1
		outputs = []
		h = self.h
		c = self.c
		for t in range(n):
			xh = np.vstack((x, h))
			fgate = sigmoid(np.dot(self.Wf, xh) + self.bf)
			igate = sigmoid(np.dot(self.Wi, xh) + self.bi)
			ogate = sigmoid(np.dot(self.Wo, xh) + self.bo)
			# generate new C
			c_new = np.tanh(np.dot(self.Wxc, xh) + self.bc)
			c = (fgate*c) + (igate*c_new)
			# generate new h
			h = ogate*np.tanh(c)
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
		xs, cs, hs, h_new, c_new, fgate, igate, ogate, probs = {}, {}, {}, {}, {}, {}, {}, {}, {}
		cs[-1] = np.copy(self.c)
		hs[-1] = np.copy(self.h)
		loss = 0
		# forward pass
		for t in range(len(inputs)):
			# encode in 1-of-k representation
			xs[t] = np.zeros((self.vocab_size,1))
			xs[t][inputs[t]] = 1
			xh = np.vstack((xs[t], hs[t-1]))
			# calculate gates
			fgate[t] = sigmoid(np.dot(self.Wf, xh) + self.bf)
			igate[t] = sigmoid(np.dot(self.Wf, xh) + self.bf)
			ogate[t] = sigmoid(np.dot(self.Wo, xh) + self.bo)
			# generate new C
			c_new[t] = np.tanh(np.dot(self.Wxc, xh) + self.bc)
			cs[t] = (fgate[t]*cs[t-1]) + (igate[t]*c_new[t])
			# generate new h
			h_new[t] = np.tanh(cs[t])
			hs[t] = ogate[t]*h_new[t]
			# map to y
			y = np.dot(self.Why, hs[t]) + self.by
			probs[t] = np.exp(y) / np.sum(np.exp(y)) # probabilities for next chars
			loss += -np.log(probs[t][targets[t],0]) # softmax (cross-entropy loss)

		# backward pass: gradients
		dWf, dbf = np.zeros_like(self.Wf), np.zeros_like(self.bf)
		dWi, dbi = np.zeros_like(self.Wi), np.zeros_like(self.bi)
		dWo, dbo = np.zeros_like(self.Wo), np.zeros_like(self.bo)
		dWxc, dbc = np.zeros_like(self.Wxc), np.zeros_like(self.bc)
		dWhy, dby = np.zeros_like(self.Why), np.zeros_like(self.by)
		
		dhnext = np.zeros_like(self.h) # derivative wrt h(t+1), since h(t) is propagated to t+1.
		dcnext = np.zeros_like(self.c) # derivative wrt c(t+1), since c(t) is propagated to t+1.
		for t in reversed(range(len(inputs))):
			# backprop cross entropy loss through softmax
			dy = np.copy(probs[t])
			dy[targets[t]] -= 1.0
			dWhy += np.dot(dy, hs[t].T)
			dby += dy

			# backprop to h & add backprop from t+1
			dh = np.dot(self.Why.T, dy) + dhnext

			xh = np.vstack((xs[t], hs[t-1]))
			# backprop through o-gate (sigmoid)
			do_raw = dh*hs[t]*(1.0 - ogate[t])
			dWo += np.dot(do_raw, xh.T)
			dbo += do_raw
			
			# backprop to c
			dc = dh*ogate[t]*(1.0 - h_new[t]*h_new[t]) + dcnext

			# backprop before i-gate and through tanh
			dc_raw = dc*igate[t]*(1.0 - c_new[t]*c_new[t])
			dWxc += np.dot(dc_raw, xh.T)
			dbc += dc_raw

			# backprop to f-gate and i-gate params
			df_raw = dc*((cs[t-1]*fgate[t]*(1.0 - fgate[t])) +  (c_new[t]*igate[t]*(1.0 - igate[t])))
			dWf += np.dot(df_raw, xh.T)
			dbf += df_raw
			
			# backprop to h(t-1) and c(t-1)
			dcnext = dc*fgate[t]
			dxh = np.dot(self.Wo.T, do_raw) + np.dot(self.Wf.T, df_raw) + np.dot(self.Wxc.T, dc_raw)

		for dparam in [dWf, dbf, dWi, dbi, dWxc, dbc, dWo, dbo, dWhy, dby]:
			np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

		self.h = hs[len(inputs)-1]
		self.c = cs[len(inputs)-1]
		# perform parameter update with Adagrad
		for param, dparam, mem in zip(
			[self.Wf, self.bf, self.Wi, self.bi, self.Wxc, self.bc, self.Wo, self.bo, self.Why, self.by], 
			[dWf, dbf, dWi, dbi, dWxc, dbc, dWo, dbo, dWhy, dby], 
			[self.mWf, self.mbf, self.mWi, self.mbi, self.mWxc, self.mbc, self.mWo, self.mbo, self.mWhy, self.mby]):
			mem += dparam * dparam
			param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

		return loss	