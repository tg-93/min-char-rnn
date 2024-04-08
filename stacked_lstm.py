# stacked_lstm.py
# an n-layer lstm, all of the same size, coded in raw pytorch, making use of automatic differetiation & Adam!
# TODO: maintain H and C over successive iterations by copying values (directly keeping track results in autograd loops) - use torch.no_grad()
# TODO: create separate classes for the single layer lstm and chain them in pytorch/tf style.
# TODO: try batch norm?

import torch
import torch.nn.functional as F
import numpy as np

class StackedLSTM:
	def __init__(self, hidden_size, vocab_size, num_layers, device = 'cpu'):
		num_params = 0
		self.num_layers = num_layers
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.device = device
		self.Ws = []
		w0 = np.random.randn(4 * hidden_size, 1 + hidden_size + vocab_size)/ (hidden_size+vocab_size) ** 0.5 # kaiming initialisation
		w0 *= 5.0/3 # kaiming init factor for tanh nonlinearity.
		w0[:,-1] = 0.0 # start with 0 bias. maybe not the best approach, but not terrible either.
		w0[:hidden_size, -1] += 1.0 # initialise forget gate to be on (aka to not forget)
		num_params += w0.size
		self.Ws.append(torch.tensor(w0, requires_grad = True, dtype=torch.float32, device=device))
		self.Wy = torch.tensor(np.random.randn(vocab_size, hidden_size) / hidden_size**0.5, requires_grad = True, dtype=torch.float32, device=device)
		self.by = torch.zeros(vocab_size, 1, requires_grad = True, dtype=torch.float32, device=device)
		num_params += self.Wy.nelement()
		for _ in range(num_layers-1):
			w = np.random.randn(4 * hidden_size, 1 + (2 * hidden_size)) / (2*hidden_size)**0.5
			w[:, -1] = 0.0
			w[:hidden_size,-1] = 1.0
			self.Ws.append(torch.tensor(w, requires_grad = True, dtype=torch.float32, device=device))
			num_params += w.size
		self.optimizer = torch.optim.Adam([self.Wy, self.by] + self.Ws)
		print(f'Number of trainable params: {num_params}')

	def training_step(self, inputs, targets, learning_rate):
		"""
		expects 2d array for inputs and targets, or shape seq_length * batch_size, i.e., each row in inputs
		should contain the char indices for timestep t for the entire batch.
		"""
		if not torch.is_tensor(inputs):
			inputs = torch.tensor(inputs, device = self.device)
		hidden_size = self.hidden_size
		batch_size = len(inputs[0])
		h_empty = torch.zeros(hidden_size, batch_size, dtype=torch.float32, device=self.device)
		c_empty = torch.zeros(hidden_size, batch_size, dtype=torch.float32, device=self.device)
		out = []
		Cs = {}
		Hs = {}
		loss = 0
		for i in range(len(inputs)):
			x = F.one_hot(inputs[i], self.vocab_size).float().T
			# print(f'input: {inputs[i]}')
			# print(f'encoded input: {x.T}')

			for j in range(self.num_layers):
				xh = torch.vstack((x if j==0 else Hs[j-1], Hs[j] if i > 0 else h_empty, 
					torch.ones(1,batch_size, dtype=torch.float32, device=self.device)))
				fioc = self.Ws[j] @ xh
				c_new = torch.tanh(fioc[3 * hidden_size :])
				Cs[j] = (Cs[j] if i > 0 else c_empty) * torch.sigmoid(fioc[:hidden_size])
				Cs[j] += c_new * torch.sigmoid(fioc[hidden_size : 2 * hidden_size])
				Hs[j] = torch.tanh(Cs[j]) * torch.sigmoid(fioc[2 * hidden_size : 3 * hidden_size])
				# print(f'h: {Hs[j].T}')
			logits = self.Wy @ Hs[self.num_layers-1] + self.by
			loss += F.cross_entropy(logits.T, torch.tensor(targets[i], device=self.device))
			# print(f'loss for input: {-torch.log(probs[targets[i]]).item()}')
		loss /= len(targets)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.item()

	def sample(self, n, seed_ix):
		hidden_size = self.hidden_size
		h_empty = torch.zeros(hidden_size, 1, dtype=torch.float32, device=self.device)
		c_empty = torch.zeros(hidden_size, 1, dtype=torch.float32, device=self.device)
		out = [seed_ix]
		Cs = {}
		Hs = {}
		for i in range(n):
			x = torch.zeros(self.vocab_size, 1, dtype=torch.float32, device=self.device)
			x[out[-1], 0] += 1

			for j in range(self.num_layers):
				W = self.Ws[j]
				xh = torch.vstack((x if j==0 else Hs[j-1], Hs[j] if i > 0 else h_empty, torch.ones(1,1, dtype=torch.float32, device=self.device)))
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