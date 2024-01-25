# stacked_lstm.py
# an n-layer lstm, all of the same size, coded in raw pytorch, making use of automatic differetiation!!

import torch
import numpy as np

class StackedLSTM:
	def __init__(self, hidden_size, vocab_size, num_layers):
		self.num_layers = num_layers
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.Ws = []
		w0 = np.random.randn(4 * hidden_size, 1 + hidden_size + vocab_size) * 0.01
		w0[:,-1] = 0.0
		w0[:hidden_size, -1] += 1.0
		self.Ws.append(torch.tensor(w0, requires_grad = True, dtype=torch.float64))
		self.Wy = torch.tensor(np.random.randn(vocab_size, hidden_size) * 0.01, requires_grad = True, dtype=torch.float64)
		self.by = torch.zeros(vocab_size, 1, requires_grad = True, dtype=torch.float64)
		for _ in range(num_layers-1):
			w = np.random.randn(4 * hidden_size, 1 + (2 * hidden_size)) * 0.01
			w[:, -1] = 0.0
			w[:hidden_size,-1] = 1.0
			self.Ws.append(torch.tensor(w, requires_grad = True, dtype=torch.float64))
		self.optimizer = torch.optim.Adam([self.Wy, self.by] + self.Ws)

	def training_step(self, inputs, targets, learning_rate):
		hidden_size = self.hidden_size
		h_empty = torch.zeros(hidden_size, 1, dtype=torch.float64)
		c_empty = torch.zeros(hidden_size, 1, dtype=torch.float64)
		out = []
		Cs = {}
		Hs = {}
		loss = 0
		for i in range(len(inputs)):
			x = torch.zeros(self.vocab_size, 1, dtype=torch.float64)
			x[inputs[i], 0] += 1
			# print(f'input: {inputs[i]}')
			# print(f'encoded input: {x.T}')

			for j in range(self.num_layers):
				xh = torch.vstack((x if j==0 else Hs[j-1], Hs[j] if i > 0 else h_empty, 
					torch.ones(1,1, dtype=torch.float64)))
				fioc = self.Ws[j] @ xh
				c_new = torch.tanh(fioc[3 * hidden_size :])
				Cs[j] = (Cs[j] if i > 0 else c_empty) * torch.sigmoid(fioc[:hidden_size])
				Cs[j] += c_new * torch.sigmoid(fioc[hidden_size : 2 * hidden_size])
				Hs[j] = torch.tanh(Cs[j]) * torch.sigmoid(fioc[2 * hidden_size : 3 * hidden_size])
				# print(f'h: {Hs[j].T}')
			logprobs = self.Wy @ Hs[self.num_layers-1] + self.by
			probs = torch.softmax(logprobs, dim=0)
			# print(f'probs for next char: {probs.T}')
			loss += -torch.log(probs[targets[i]])
			# print(f'loss for input: {-torch.log(probs[targets[i]]).item()}')
		# loss /= len(targets)
		loss.backward()
		self.optimizer.step()
		self.optimizer.zero_grad()
		return loss.item()/len(targets)

	def sample(self, n, seed_ix):
		hidden_size = self.hidden_size
		h_empty = torch.zeros(hidden_size, 1, dtype=torch.float64)
		c_empty = torch.zeros(hidden_size, 1, dtype=torch.float64)
		out = [seed_ix]
		Cs = {}
		Hs = {}
		for i in range(n):
			x = torch.zeros(self.vocab_size, 1, dtype=torch.float64)
			x[out[-1], 0] += 1

			for j in range(self.num_layers):
				W = self.Ws[j]
				xh = torch.vstack((x if j==0 else Hs[j-1], Hs[j] if i > 0 else h_empty, torch.ones(1,1, dtype=torch.float64)))
				fioc = W @ xh
				c_new = torch.tanh(fioc[3 * hidden_size :])
				Cs[j] = ((Cs[j] if i > 0 else c_empty) * torch.sigmoid(fioc[:hidden_size])) + (c_new * torch.sigmoid(fioc[hidden_size : 2 * hidden_size]))
				Hs[j] = torch.tanh(Cs[j]) * torch.sigmoid(fioc[2 * hidden_size : 3 * hidden_size])
			logprobs = (self.Wy @ Hs[self.num_layers-1]) + self.by
			probs = torch.softmax(logprobs, dim=0)
			sample = torch.multinomial(probs.T, 3)
			# print(sample)
			out.append(sample[0,0].item())
		return out

	def reset_memory(self):
		return

	def save(self, name):
		return