# a simple, non-lstm rnn

import numpy as np

def lossFun(inputs, targets, hprev, vocab_size, Wxh, Whh, Why, bh, by):
	"""
  	inputs,targets are both list of integers.
  	hprev is Hx1 array of initial hidden state
  	returns the loss, gradients on model parameters, and last hidden state
  	"""
  	xs, hs, ys, probs = {}, {}, {}, {}
  	hs[-1] = np.copy(hprev)
  	loss = 0
  	# forward pass
  	for t in xrange(len(inputs)):
    	xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    	xs[t][inputs[t]] = 1
    	hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    	ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    	probs[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    	loss += -np.log(probs[t][targets[t],0]) # softmax (cross-entropy loss)

    # backward pass: gradients
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0]) # derivative wrt h(t+1), since h(t) affects y(t+1) also.
    for t in reversed(xrange(len(inputs))):
    	dy = np.copy(probs[t])
    	dy[targets[t]] -= 1 # softmax derivative
    	dWhy += np.dot(dy, hs[t].T)
	    dby += dy
	    dh = np.dot(Why.T, dy) + dhnext # backprop into h
	    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
	    dbh += dhraw
	    dWxh += np.dot(dhraw, xs[t].T)
	    dWhh += np.dot(dhraw, hs[t-1].T)
	    dhnext = np.dot(Whh.T, dhraw)
	
	for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    	np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  	
  	return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(n, seed_ix, hprev, vocab_size, Wxh, Whh, Why, bh, by):
	""" 
  	sample a sequence of integers from the model 
  	h is memory state, seed_ix is seed letter for first time step
  	"""
  	x = np.zeros((vocab_size, 1))
  	x[seed_ix] = 1
  	outputs = []
  	for t in xrange(n):
    	h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
	    y = np.dot(Why, h) + by
	    p = np.exp(y) / np.sum(np.exp(y))
	    ix = np.random.choice(range(vocab_size), p=p.ravel())
	    x = np.zeros((vocab_size, 1))
	    x[ix] = 1
	    outputs.append(ix)
  	return outputs
