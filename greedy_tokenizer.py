# greedy_tokenizer.py
# Simple Byte Pair Encoding implementation that directly encodes characters 
# to tokens and vice-versa, without going into the bytestream details.
# This one is a bit greedy, and chucks out infrequent tokens once they've been used
# up by larger tokens. for example, if the tokenizer learnt a token for "tok", but later
# learnt another token for "token", such that "tok" is no longer a frequent token by itself,
# it will free up the token id for "tok" and add something else to the vocab instead.

import torch
import torch.nn.functional as F
import numpy as np
import math
import collections

def add_deps(deps, token, ngram):
	for x in ngram:
		if x in deps:
			deps[x].append(token)
		else:
			deps[x] = [token]

class GreedyTokenizer:
	def __init__(self, text: str, max_vocab_size: int, token_min_freq: int, skip_chars = [' ', ',', '.', '\n', '\t']):
		# generate encoding and decoding tables from text
		self.chars = sorted(list(set(text)))
		self.word_to_token = { ch:i for i,ch in enumerate(self.chars) }
		self.token_to_word = { i:ch for i,ch in enumerate(self.chars) }

		max_vocab_size = min(max_vocab_size, math.sqrt(len(text)))
		encoded_text = [self.word_to_token[ch] for ch in text]
		self.token_expansion = {} 	# token to sequence of smaller tokens
		self.inv_index = {} 		# token to list of tokens that rely on this token
		self.next_token = len(self.word_to_token)
		self.skip_tokens = set([self.word_to_token.get(x, -1) for x in skip_chars])
		while self.next_token < max_vocab_size:
			if self.next_token % 10 == 0:
				print("Vocab Size: ", self.next_token)
				print("Text Length: ", len(encoded_text))
			# compute bigram freqs
			bigram_freqs = collections.Counter(self.get_bigrams(encoded_text))
			# print("New Bigram count: ", len(bigram_freqs))
			if len(bigram_freqs) == 0:
				break
			# Find most frequent bigram
			top_bigram = max(bigram_freqs, key=bigram_freqs.get)
			if bigram_freqs[top_bigram] < token_min_freq:
				break
			# Add most frequent bigram to tokens and index
			self.token_expansion[self.next_token] = top_bigram
			add_deps(self.inv_index, self.next_token, top_bigram)
			# unroll token for faster decoding
			token_text = self.unroll(self.next_token)
			self.word_to_token[token_text] = self.next_token
			self.token_to_word[self.next_token] = token_text
			# update text
			encoded_text = self.compress_once(encoded_text, top_bigram, self.next_token)
			self.next_token += 1
		# remove the most infrequent token and see if it is still the token that gets added
		while True:
			# find next bigram to encode
			# compute bigram freqs
			bigram_freqs = collections.Counter(self.get_bigrams(encoded_text))
			if len(bigram_freqs) == 0:
				break
			top_bigram = max(bigram_freqs, key=bigram_freqs.get)
			# find most infrequent unigram (token)
			unigram_freqs = collections.Counter(encoded_text)
			popped_token = min(unigram_freqs, key=lambda x: unigram_freqs[x] if x >= len(self.chars) else float('inf'))
			# Only swap if new bigram is at least 3x more frequent. This means that a token cannot be popped to add in 
			# a compound token that contains it.
			if bigram_freqs[top_bigram] < 3 * unigram_freqs[popped_token]:
				print("incumbent freq: ", unigram_freqs[popped_token], "challenger freq: ", bigram_freqs[top_bigram])
				for incumbent in unigram_freqs:
					if incumbent >= len(self.chars):
						print(self.token_to_word[incumbent], " : ", unigram_freqs[incumbent])
				break
			print("Replacing token: ", self.token_to_word[popped_token])
			# find compound tokens containing popped token from inv_index
			compound_tokens = self.inv_index[popped_token]
			# for popped token's expansion, add each compound token to their inv_index
			popped_token_expansion = self.token_expansion[popped_token]
			for t in popped_token_expansion:
				self.inv_index[t].remove(popped_token)
				self.inv_index[t].extend(compound_tokens)
			# for each compound token, update its expansion in token_expansion
			for ct in compound_tokens:
				new_expansion = []
				for t in self.token_expansion[ct]:
					if t == popped_token:
						new_expansion.extend(popped_token_expansion)
					else:
						new_expansion.append(t)
				self.token_expansion[ct] = new_expansion
			# expand encoded text with popped token's expansion
			encoded_text = self.decompress_once(encoded_text, popped_token_expansion, popped_token)
			# remove popped token from word_to_token and token_to_word
			del self.word_to_token[self.token_to_word[popped_token]]
			del self.token_to_word[popped_token]
			# remove popped token from token_expansion and inv_index
			del self.inv_index[popped_token]
			del self.token_expansion[popped_token]
			# TODO: add top_bigram similar to previous loop
			# Add most frequent bigram to tokens and index
			self.token_expansion[popped_token] = top_bigram
			add_deps(self.inv_index, popped_token, top_bigram)
			# unroll token for faster decoding
			token_text = self.unroll(popped_token)
			print("Replaced with: ", token_text)
			self.word_to_token[token_text] = popped_token
			self.token_to_word[popped_token] = token_text
			# update text
			encoded_text = self.compress_once(encoded_text, top_bigram, popped_token)
		self.print_codebook()

	def print_codebook(self):
		ans = []
		for token in self.token_expansion:
			ans.append(self.unroll(token))
		print(ans)

	def unroll(self, token: int) -> str:
		if token not in self.token_expansion and token not in self.token_to_word:
			return "===== UNKNOWN ====="
		if token in self.token_to_word:
			return self.token_to_word[token]
		token_list = []
		expanded = [token]
		while len(expanded) > len(token_list):
			token_list = expanded
			expanded = []
			for x in token_list:
				if x in self.token_expansion:
					expanded.extend(self.token_expansion[x])
				else:
					expanded.append(x)
		ret = "".join([self.token_to_word[x] for x in expanded])
		return ret


	def get_bigrams(self, text: list[int]):
		ret = []
		for bigram in zip(text, text[1:]):
			if (bigram[0] not in self.skip_tokens) and (bigram[1] not in self.skip_tokens):
				ret.append(bigram)
		return ret

	def decompress_once(self, encoded_text: list[int], expansion: list[int], token: int) -> list[int]:
		# takes an encoded text and a token, and replaces all its occurences with expansion
		ret = []
		for x in encoded_text:
			if x == token:
				ret.extend(expansion)
			else:
				ret.append(x)
		return ret

	def compress_once(self, encoded_text: list[int], bigram: tuple[int], token: int) -> list[int]:
		# takes an encoded text and a bigram, and replaces all its occurences with token
		ret = []
		i = 0
		while i < len(encoded_text)-1:
			curr_bigram = (encoded_text[i], encoded_text[i+1])
			if curr_bigram == bigram:
				ret.append(token)
				i += 2
			else:
				ret.append(encoded_text[i])
				i += 1
		if i < len(encoded_text):
			ret.append(encoded_text[i])
		return ret

	def encode(self, text: str) -> list[int]:
		encoded = []
		max_token_len = len(max(self.word_to_token, key=len))
		i = 0
		while i < len(text):
			for j in range(max_token_len, 0, -1):
				if i + j >= len(text):
					continue
				maybe_token = text[i: i+j]
				if maybe_token in self.word_to_token:
					encoded.append(self.word_to_token[maybe_token])
					break
			i += j
		return encoded

	def decode(self, tokens: list[int]) -> str:
		ans = []
		for i in tokens:
			ans.append(self.token_to_word[i] if i in self.token_to_word else "<UNKOWN>")
		return "".join(ans)

# data I/O
data = open('wot1.txt', 'r').read()
my_bpe = GreedyTokenizer(data, 500, 50)