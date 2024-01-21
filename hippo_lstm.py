# Hippo based LSTM with no peepholes. Input and forget gates are coupled, as in the hippo paper.
# Instead of generating h[t] via the output gate and tanh on C[t],
# we do it by adding a linear layer on C[t] and then keeping track of it using hippo
# hippo output is h[t]

import numpy as np
import os
import scipy.linalg as la
import scipy.special as sp
from datetime import datetime

def transition_mats(N):
	q = np.arange(N, dtype=np.float64)
	col, row = np.meshgrid(q, q)
	r = 2 * q + 1
	M = -(np.where(row >= col, r, 0) - np.diag(q))
	T = np.sqrt(np.diag(2 * q + 1))
	A = np.dot(np.dot(T, M), np.linalg.inv(T))
	B = np.diag(T)[:, None]
	return A, B

class HippoLSTM:
	def __init__(self, hidden_size, vocab_size, feature_size):
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.feature_size = feature_size
		self.memory_size = hidden_size//feature_size
		self.c = np.zeros((hidden_size, 1))
		self.h = np.zeros((hidden_size, 1))
		## Model params
		# input gate
		self.Wi = np.random.randn(hidden_size, vocab_size+hidden_size)*0.01
		self.bi = np.zeros((hidden_size, 1))
		# C update
		self.Wc = np.random.randn(hidden_size, vocab_size+hidden_size)*0.01
		self.bc = np.zeros((hidden_size, 1))
		# generate y from C
		self.Wy = np.random.randn(vocab_size, hidden_size) * 0.01
		self.by = np.zeros((vocab_size, 1))
		# generate l from C
		self.Wl = np.random.randn(feature_size, hidden_size) * 0.01
		self.bl = np.zeros((feature_size, 1))
		## Fixed params for hippo
		self.A, self.B = transition_mats(self.memory_size)
		# memory variables for Adagrad
		self.mWi, self.mbi = np.zeros_like(self.Wi), np.zeros_like(self.bi)
		self.mWc, self.mbc = np.zeros_like(self.Wc), np.zeros_like(self.bc)
		self.mWy, self.mby = np.zeros_like(self.Wy), np.zeros_like(self.by)
		self.mWl, self.mbl = np.zeros_like(self.Wl), np.zeros_like(self.bl)

	def reset_memory(self):
		if self.c is not None:
			self.c = np.zeros_like(self.c)
		if self.h is not None:
			self.h = np.zeros_like(self.h)

	def save(self, name):
		current_date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
		name = f'simple_lstm_{name}_{current_date_time}'
		os.mkdir(name)
		np.save(name + '/Wl.npy', self.Wl)
		np.save(name + '/Wi.npy', self.Wi)
		np.save(name + '/Wc.npy', self.Wc)
		np.save(name + '/Wy.npy', self.Wy)
		np.save(name + '/bl.npy', self.bl)
		np.save(name + '/bi.npy', self.bi)
		np.save(name + '/bc.npy', self.bc)
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
		c = self.c
		for t in range(n):
			xh = np.vstack((x, h))
			igate = sp.expit(np.dot(self.Wi, xh) + self.bi)
			# generate new C
			c_new = np.tanh(np.dot(self.Wc, xh) + self.bc)
			c = ((1.0-igate)*c) + (igate*c_new)
			y = np.dot(self.Wy, c) + self.by
			probs = sp.softmax(y)
			ix = np.random.choice(range(self.vocab_size), p=probs.ravel())
			x = np.zeros_like(x)
			x[ix] = 1
			outputs.append(ix)
			# generate new h
			l = np.dot(self.Wl, c)
			h, _, _ = self.hippo_update(h.reshape(self.memory_size, self.feature_size), l, t+1)
			h = h.reshape(self.hidden_size, 1)
		return outputs

	def hippo_update(self, h_prev, l, t):
		A1 = la.inv(np.eye(self.memory_size) - 0.5*self.A/(t+1))
		A2 = np.eye(self.memory_size) + 0.5*self.A/t
		Bt = self.B/t
		At = np.dot(A1, A2)
		return np.dot(At, h_prev) + Bt*l.T, At, Bt

	def training_step(self, inputs, targets, learning_rate):
		"""
		inputs,targets are both list of integers.
		returns the loss, gradients on model parameters, and last hidden state
		"""
		xs, cs, hs, c_new, igate, probs, a, b = {}, {}, {}, {}, {}, {}, {}, {}
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
			igate[t] = sp.expit(np.dot(self.Wi, xh) + self.bi)
			# generate new C
			c_new[t] = np.tanh(np.dot(self.Wc, xh) + self.bc)
			cs[t] = ((1.0-igate[t])*cs[t-1]) + (igate[t]*c_new[t])
			# map to y
			y = np.dot(self.Wy, cs[t]) + self.by
			probs[t] = sp.softmax(y) # probabilities for next chars
			loss += -np.log(probs[t][targets[t],0]) # softmax (cross-entropy loss)
			# generate new h
			l = np.dot(self.Wl, cs[t])
			ht, a[t], b[t] = self.hippo_update(hs[t-1].reshape(self.memory_size, self.feature_size), l, t+1)
			hs[t] = ht.reshape(self.memory_size*self.feature_size, 1)

		# backward pass: gradients
		dWi, dbi = np.zeros_like(self.Wi), np.zeros_like(self.bi)
		dWc, dbc = np.zeros_like(self.Wc), np.zeros_like(self.bc)
		dWy, dby = np.zeros_like(self.Wy), np.zeros_like(self.by)
		dWl, dbl = np.zeros_like(self.Wl), np.zeros_like(self.bl)
		
		dhnext = np.zeros_like(self.h) # derivative wrt h(t+1), since h(t) is propagated to t+1.
		dcnext = np.zeros_like(self.c) # derivative wrt c(t+1), since c(t) is propagated to t+1.
		for t in reversed(range(len(inputs))):
			# backprop cross entropy loss through softmax
			dy = np.copy(probs[t])
			dy[targets[t]] -= 1.0
			dWy += np.dot(dy, cs[t].T)
			dby += dy

			# dh is only dhnext in this case, since y depends on c instead of h
			# backprop dh via hippo
			dh = dhnext.reshape(self.memory_size, self.feature_size)
			dl = np.dot(dh.T, b[t])
			dWl += np.dot(dl, cs[t].T)
			dbl += dl

			# dc will have 3 components: dy, dcnext and dl
			dc = np.dot(self.Wy.T, dy) + dcnext
			dc += np.dot(self.Wl.T, dl)

			xh = np.vstack((xs[t], hs[t-1]))

			# backprop before i-gate and through tanh
			dc_raw = dc*igate[t]*(1- c_new[t]*c_new[t])
			dWc += np.dot(dc_raw, xh.T)
			dbc += dc_raw

			# backprop to f-gate and i-gate params
			di_raw = dc*(c_new[t] - cs[t-1])*igate[t]*(1.0 - igate[t])
			dWi += np.dot(di_raw, xh.T)
			dbi += di_raw
			
			# backprop to h(t-1) and c(t-1)
			dcnext = dc*(1.0 - igate[t])
			dxh = np.dot(self.Wc.T, dc_raw) + np.dot(self.Wi.T, di_raw)

			# dhnext will have 4 components: hippo, c_new[t], input/forget gates
			dhnext = np.dot(a[t].T, dh).reshape(self.memory_size*self.feature_size, 1)
			dhnext += dxh[self.vocab_size:]

		for dparam in [dWi, dbi, dWc, dbc, dWl, dbl, dWy, dby]:
			np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

		self.h = hs[len(inputs)-1]
		self.c = cs[len(inputs)-1]
		# perform parameter update with Adagrad
		for param, dparam, mem in zip(
			[self.Wi, self.bi, self.Wc, self.bc, self.Wl, self.bl, self.Wy, self.by], 
			[dWi, dbi, dWc, dbc, dWl, dbl, dWy, dby], 
			[self.mWi, self.mbi, self.mWc, self.mbc, self.mWl, self.mbl, self.mWy, self.mby]):
			mem += dparam * dparam
			param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

		return loss	