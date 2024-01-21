# Hippo RNN
# A minimalist implementation of the hippo update rule in an RNN.
# This deviates significantly from the original implementation (https://arxiv.org/pdf/2008.07669.pdf)
# by (1) not having any gates, and 
#    (2) only propagating a single vector (h) through time, like a classic RNN.
# Since Whh and Wxh in Hippo are fixed based on the measure function, the only 
# learned params would be Why and by.
# However, Hippo keeps track of a single 1D variable over time, so we add an 
# intermediate vectpr of 1d "features" of x called f, produced by a linear transform + a tanh.
# h tracks the history of f via hippo updates and h is linearly tranformed to y as before.
# Number of learnable params: O(hidden*vocab + feature*vocab)

import numpy as np
import scipy as sp
import math
import os
from datetime import datetime

class HippoRNN:
	def __init__(self, feature_size, memory_size, vocab_size, peephole = False, hprev = None):
		print(f'Starting Hippo RNN with feature_size: {feature_size}, memory_size: {memory_size}, and vocab size: {vocab_size}')
		self.vocab_size = vocab_size
		hidden_size = feature_size*memory_size
		self.hidden_size = hidden_size
		self.feature_size = feature_size # number of features to keep track of in hidden layer
		self.memory_size = memory_size # memory size per feature
		self.peephole = peephole
		# self.h = hprev if hprev else np.zeros((memory_size, feature_size))
		# static params
		self.Whh_fixed = np.zeros((memory_size, memory_size), dtype=np.float64)
		for i in range(memory_size):
			for j in range(i+1):
				self.Whh_fixed[i][j] -= math.sqrt(2*i + 1)*math.sqrt(2*j+1) if i > j else i+1 
		self.Wfh_fixed = np.sqrt(np.arange(1.0, 2.0*memory_size +1.0, 2.0, dtype=np.float64)).reshape((memory_size, 1))
		print(self.Whh_fixed)
		print(self.Wfh_fixed)
		# Model params
		self.Wxf = np.random.randn(feature_size, vocab_size)*0.01 # input to hidden
		self.bf = np.zeros((feature_size,1))
		self.Why = np.random.randn(vocab_size, hidden_size)*0.1 # hidden to output
		self.by = np.zeros((vocab_size, 1)) # output bias
		if peephole:
			self.Wfy = np.random.randn(vocab_size, feature_size)*0.1
			print(f'size of trainable params: {self.Wxf.size + self.bf.size + self.Why.size + self.by.size + self.Wfy.size}')
		else:
			self.Wfy = np.zeros((vocab_size, feature_size))
			print(f'size of trainable params: {self.Wxf.size + self.bf.size + self.Why.size + self.by.size}')
		# memory variables for Adagrad
		self.mWxf, self.mWhy = np.zeros_like(self.Wxf), np.zeros_like(self.Why)
		self.mbf, self.mby = np.zeros_like(self.bf), np.zeros_like(self.by)
		self.mWfy = np.zeros_like(self.Wfy)

	def transition_mats(self, N):
		q = np.arange(N, dtype=np.float64)
		col, row = np.meshgrid(q, q)
		r = 2 * q + 1
		M = -(np.where(row >= col, r, 0) - np.diag(q))
		T = np.sqrt(np.diag(2 * q + 1))
		A = np.dot(np.dot(T, M), np.linalg.inv(T))
		B = np.diag(T)[:, None]
		return A, B

	def reset_memory(self):
		# if self.h is not None:
		# 	self.h = np.zeros_like(self.h)
		return

	def save(self, name):
		current_date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
		name = f'hippo_rnn_{name}_{current_date_time}'
		os.mkdir(name)
		np.save(name + '/Wxf.npy', self.Wxf)
		np.save(name + '/Why.npy', self.Why)
		np.save(name + '/bf.npy', self.bf)
		np.save(name + '/by.npy', self.by)
		if self.peephole:
			np.save(name + '/Wfy.npy', self.Wfy)

	def elongate(self, h):
		return h.reshape((h.shape[0]*h.shape[1],1))

	def sample(self, n, seed_ix):
		""" 
		sample a sequence of integers from the model 
		hprev is memory state, seed_ix is seed letter for first time step
		"""
		x = np.zeros((self.vocab_size, 1))
		x[seed_ix] = 1
		outputs = []
		h = np.zeros((self.memory_size, self.feature_size))
		for t in range(n):
			f = np.tanh(np.dot(self.Wxf, x) + self.bf)
			A1 = np.linalg.inv(np.eye(self.memory_size) - 0.5*(self.Whh_fixed/(t+2.0)))
			A2 = np.eye(self.memory_size) + 0.5*(self.Whh_fixed/(t+1.0))
			A = np.dot(A1, A2)
			B = np.dot(A1, self.Wfh_fixed / (t+1.0))
			h = np.dot(A, h) + (B * f.T)
			if self.peephole:
				y = np.dot(self.Why, self.elongate(h)) + np.dot(self.Wfy, f)+ self.by
			else:
				y = np.dot(self.Why, self.elongate(h)) + self.by
			probs = sp.special.softmax(y)
			ix = np.random.choice(range(self.vocab_size), p=probs.ravel())
			x = np.zeros_like(x)
			x[ix] = 1
			outputs.append(ix)
			
		# print(f'Feature percentiles: {np.percentile(f, [10, 30, 50, 70, 90])}')
		# print(f'Wf percentiles: {np.percentile(self.Wxf, [10, 30, 50, 70, 90])}')
		# print(f'Hippo memory percentiles: {np.percentile(h, [10, 30, 50, 70, 90])}')
		# print(f'Wy percentiles: {np.percentile(self.Why, [10, 30, 50, 70, 90])}')
		# print(f'max prob: {np.max(probs)} for char_index: {np.argmax(probs)}')
		return outputs

	def training_step(self, inputs, targets, learning_rate):
		"""
		inputs,targets are both list of integers.
		returns the loss, gradients on model parameters, and last hidden state
		"""
		xs, fs, hs, probs = {}, {}, {}, {}
		hs[-1] = np.zeros((self.memory_size, self.feature_size))
		loss = 0
		# forward pass
		for t in range(len(inputs)):
			xs[t] = np.zeros((self.vocab_size,1)) # encode in 1-of-k representation
			xs[t][inputs[t]] = 1
			fs[t] = np.tanh(np.dot(self.Wxf, xs[t]) + self.bf)
			A1 = np.linalg.inv(np.eye(self.memory_size) - 0.5*(self.Whh_fixed/(t+2.0)))
			A2 = np.eye(self.memory_size) + 0.5*(self.Whh_fixed/(t+1.0))
			A = np.dot(A1, A2)
			B = np.dot(A1, self.Wfh_fixed / (t+1.0))
			hs[t] = np.dot(A, hs[t-1]) + (B * fs[t].T)
			# print(hs[t])
			y = np.dot(self.Why, self.elongate(hs[t])) + self.by # unnormalized log probabilities for next chars
			if self.peephole:
				y += np.dot(self.Wfy, fs[t])
			# print(y)
			probs[t] = sp.special.softmax(y) # probabilities for next chars
			loss += -np.log(probs[t][targets[t],0]) # softmax (cross-entropy loss)

		# backward pass: gradients
		dWxf, dWhy, dWfy = np.zeros_like(self.Wxf), np.zeros_like(self.Why), np.zeros_like(self.Wfy)
		dbf, dby = np.zeros_like(self.bf), np.zeros_like(self.by)
		dhnext = np.zeros_like(hs[0]) # derivative wrt h(t+1), since h(t) affects y(t+1) also.
		for t in reversed(range(len(inputs))):
			dy = np.copy(probs[t])
			dy[targets[t]] -= 1 # derivative of negative log of softmax
			dWhy += np.dot(dy, self.elongate(hs[t]).T)
			dby += dy
			dh = np.dot(self.Why.T, dy).reshape((self.memory_size, self.feature_size)) # backprop into h
			dh += dhnext
			A1 = np.linalg.inv(np.eye(self.memory_size) - 0.5*(self.Whh_fixed/(t+2.0)))
			A2 = np.eye(self.memory_size) + 0.5*(self.Whh_fixed/(t+1.0))
			A = np.dot(A1, A2)
			B = np.dot(A1, self.Wfh_fixed / (t+1.0))
			df = np.dot(dh.T, B)
			if self.peephole:
				dWfy += np.dot(dy, fs[t].T)
				df += np.dot(self.Wfy.T, dy)
			df_raw = df*(1 - fs[t]*fs[t])
			dWxf += np.dot(df_raw, xs[t].T)
			dbf += df_raw
			dhnext = np.dot(A.T, dh)

		for dparam in [dWxf, dWhy, dbf, dby, dWfy]:
			np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

		# self.h = hs[len(inputs)-1]
		# perform parameter update with Adagrad
		for param, dparam, mem in zip([self.Wxf, self.Why, self.bf, self.by, self.Wfy], 
			[dWxf, dWhy, dbf, dby, dWfy], 
			[self.mWxf, self.mWhy, self.mbf, self.mby, self.mWfy]):
			mem += dparam * dparam
			param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

		return loss