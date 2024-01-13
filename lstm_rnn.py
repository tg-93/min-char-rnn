# LSTM with no peepholes, and coupled input and forget gates

import numpy as np
import os
from datetime import datetime

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

class LSTM:
	def __init__(self, hidden_size, vocab_size, hprev = None):
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.h = hprev if hprev else np.zeros((hidden_size,1))
		## Model params
		# forget gate
		self.Wf = np.random.randn(hidden_size, vocab_size)*0.01
		self.bf = np.zeros((hidden_size, 1)) - 0.01 # forget gate should bias towards "off"
		# h update
		self.Wxh = np.random.randn(hidden_size, vocab_size)*0.01
		self.bh = np.zeros((hidden_size, 1))
		# output gate
		self.Wo = np.random.randn(vocab_size, vocab_size)*0.01
		self.bo = np.zeros((vocab_size, 1))
		# y generation
		self.Why = np.random.randn(vocab_size, hidden_size) * 0.01
		self.by = np.zeros((vocab_size, 1))
		# memory variables for Adagrad
		self.mWf, self.mbf = np.zeros_like(self.Wf), np.zeros_like(self.bf)
		self.mWxh, self.mbh = np.zeros_like(self.Wxh), np.zeros_like(self.bh)
		self.mWo, self.mbo = np.zeros_like(self.Wo), np.zeros_like(self.bo)
		self.mWhy, self.mby = np.zeros_like(self.Why), np.zeros_like(self.by)

	def reset_memory(self):
		if self.h is not None:
			self.h = np.zeros_like(self.h)

	def save(self, name):
		current_date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
		name = name + '_' + current_date_time
		os.mkdir(name)
		np.save(name + '/Wf.npy', self.Wf)
		np.save(name + '/Wxh.npy', self.Wxh)
		np.save(name + '/Wo.npy', self.Wo)
		np.save(name + '/Why.npy', self.Why)
		np.save(name + '/bf.npy', self.bf)
		np.save(name + '/bo.npy', self.bo)
		np.save(name + '/bh.npy', self.bh)
		np.save(name + '/by.npy', self.by)

	def forward(self, input_ix, target):
		x = np.zeros((self.vocab_size, 1))
		x[input_ix] = 1
		fgate = sigmoid(np.dot(self.Wf, x) + self.bf)
		h_new = np.tanh(np.dot(self.Wxh, x) + self.bh)
		self.h = (fgate*self.h) + (1.0-fgate)*h_new
		ogate = sigmoid(np.dot(self.Wo, x) + self.bo)
		y_a = np.dot(self.Why, self.h) + self.by
		y = ogate*(np.tanh(y_a))
		probs = np.exp(y) / np.sum(np.exp(y))
		loss = -np.log(probs[target]) # softmax (cross-entropy loss)
		return loss, probs, np.random.choice(range(self.vocab_size), p=probs.ravel())

	def sample(self, n, seed_ix):
		""" 
		sample a sequence of integers from the model 
		seed_ix is seed letter for first time step
		"""
		outputs = []
		for t in range(n):
			_, probs, y = self.forward(seed_ix, seed_ix)
			outputs.append(y)
			seed_ix = y
		return outputs

	def training_step(self, inputs, targets, learning_rate):
		"""
		inputs,targets are both list of integers.
		returns the loss, gradients on model parameters, and last hidden state
		"""
		xs, hs, h_news, ys, y_as, fgate, ogate, probs = {}, {}, {}, {}, {}, {}, {}, {}
		hs[-1] = np.copy(self.h)
		loss = 0
		# forward pass
		for t in range(len(inputs)):
			xs[t] = np.zeros((self.vocab_size,1)) # encode in 1-of-k representation
			xs[t][inputs[t]] = 1
			fgate[t] = sigmoid(np.dot(self.Wf, xs[t]) + self.bf)
			h_news[t] = np.tanh(np.dot(self.Wxh, xs[t]) + self.bh)
			hs[t] = (fgate[t]*hs[t-1]) + (1.0-fgate[t])*h_news[t]
			ogate[t] = sigmoid(np.dot(self.Wo, xs[t]) + self.bo)
			y_as[t] = np.tanh(np.dot(self.Why, hs[t]) + self.by)
			ys[t] = ogate[t]*y_as[t]
			probs[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
			loss += -np.log(probs[t][targets[t],0]) # softmax (cross-entropy loss)

		# backward pass: gradients
		dWf, dbf = np.zeros_like(self.Wf), np.zeros_like(self.bf)
		dWxh, dbh = np.zeros_like(self.Wxh), np.zeros_like(self.bh)
		dWo, dbo = np.zeros_like(self.Wo), np.zeros_like(self.bo)
		dWhy, dby = np.zeros_like(self.Why), np.zeros_like(self.by)
		dhnext = np.zeros_like(self.h) # derivative wrt h(t+1), since h(t) affects y(t+1) also.
		for t in reversed(range(len(inputs))):
			dy = np.copy(probs[t])
			dy[targets[t]] -= 1.0 # softmax derivative
			dya_raw = dy*ogate[t]*(1.0 - y_as[t]*y_as[t])
			dWhy += np.dot(dya_raw, hs[t].T)
			dby += dya_raw

			dogate_raw = dy*ys[t]*(1.0 - ogate[t])
			dWo += np.dot(dogate_raw, xs[t].T)
			dbo += dogate_raw

			dh = dhnext + np.dot(self.Why.T, dya_raw)
			dh_new_raw = dh*(1.0 - fgate[t])*(1.0 - h_news[t]*h_news[t])
			dWxh += np.dot(dh_new_raw, xs[t].T)
			dbh += dh_new_raw

			dfgate_raw = dh*fgate[t]*(1.0 - fgate[t])*(hs[t-1] - h_news[t])
			dWf += np.dot(dfgate_raw, xs[t].T)
			dbf += dfgate_raw

			dhnext = dh*fgate[t]

		for dparam in [dWf, dbf, dWxh, dbh, dWo, dbo, dWhy, dby]:
			np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

		self.h = hs[len(inputs)-1]
		# perform parameter update with Adagrad
		for param, dparam, mem in zip(
			[self.Wf, self.bf, self.Wxh, self.bh, self.Wo, self.bo, self.Why, self.by], 
			[dWf, dbf, dWxh, dbh, dWo, dbo, dWhy, dby], 
			[self.mWf, self.mbf, self.mWxh, self.mbh, self.mWo, self.mbo, self.mWhy, self.mby]):
			mem += dparam * dparam
			param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

		return loss	