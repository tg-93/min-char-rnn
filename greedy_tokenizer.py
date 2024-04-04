# greedy_tokenizer.py
# Simple Byte Pair Encoding implementation that directly encodes characters 
# to tokens and vice-versa, without going into the bytestream details.
# This one is a bit greedy, and chucks out infrequent tokens once they've been used
# up by larger tokens. for example, if the tokenizer learnt a token for "tok", but later
# learnt another token for "token", such that "tok" is no longer a frequent token by itself,
# it will free up the token id for "tok" and add something else to the vocab instead.

# TODO: use trie for faster/cleaner encoding.

import torch
import torch.nn.functional as F
import numpy as np
import math
import collections

def get_bigrams(text: list[int], skip_tokens: set[int]) -> list[(int, int)]:
	ret = []
	for bigram in zip(text, text[1:]):
		if (bigram[0] not in skip_tokens) and (bigram[1] not in skip_tokens):
			ret.append(bigram)
	return ret

def get_top_bigram(encoded_text, skip_tokens):
	bigram_freqs = collections.Counter(get_bigrams(encoded_text, skip_tokens))
	# Find most frequent bigram
	top_bigram = max(bigram_freqs, key=bigram_freqs.get)
	return top_bigram, bigram_freqs[top_bigram]

class GreedyTokenizer:
	def __init__(self, text: str, max_vocab_size: int, token_min_freq: int, skip_chars = [' ', ',', '.', '\n', '\t', '"']):
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

		encoded_text = self.build_vocab(max_vocab_size, encoded_text, token_min_freq)
		self.vocab_swapping(encoded_text, token_min_freq)
		self.print_codebook()

	def build_vocab(self, max_vocab_size: int, encoded_text: list[int], token_min_freq):
		while self.next_token < max_vocab_size:
			if self.next_token % 10 == 0:
				print("Vocab Size:", self.next_token)
				print("Text Length:", len(encoded_text))
			top_bigram, top_bigram_freq = get_top_bigram(encoded_text, self.skip_tokens)
			if top_bigram_freq < token_min_freq:
				break
			self.add_token(self.next_token, top_bigram)
			encoded_text = self.compress_once(encoded_text, top_bigram, self.next_token)
			self.next_token += 1
		return encoded_text

	def vocab_swapping(self, encoded_text: list[int], token_min_freq):
		# remove infrequent tokens to free up space for newer compound tokens.
		swap_count = 0
		while True:
			top_bigram, top_bigram_freq = get_top_bigram(encoded_text, self.skip_tokens)
			if top_bigram_freq < token_min_freq:
				print("No frequent bigram found.")
				break
			unigram_freqs = collections.Counter(encoded_text)
			popped_token = min(self.token_expansion, key=lambda x: unigram_freqs.get(x, 0) if x >= len(self.chars) else float('inf'))
			if popped_token < len(self.chars):
				print("No infrequent token found for swapping.")
				break
			if 4 * top_bigram_freq < 5 * unigram_freqs[popped_token]:
				print("incumbent:", self.token_to_word[popped_token], "freq:", unigram_freqs[popped_token], "challenger freq:", top_bigram_freq)
				break
			print("Replacing token:", self.token_to_word[popped_token])
			encoded_text = self.decompress_once(encoded_text, popped_token)
			self.remove_token(popped_token)
			self.add_token(popped_token, top_bigram)
			print("Replaced with:", self.token_to_word[popped_token])
			swap_count += 1
			encoded_text = self.compress_once(encoded_text, top_bigram, popped_token)
		print("Swapped", swap_count, "tokens!")
		print("Encoded text length:", len(encoded_text))

	def add_deps(self, token, ngram):
		for x in ngram:
			if x in self.inv_index:
				self.inv_index[x].append(token)
			else:
				self.inv_index[x] = [token]

	def add_token(self, token, expansion):
		self.token_expansion[token] = expansion
		self.add_deps(token, expansion)
		# unroll token for faster decoding
		token_text = self.unroll(token)
		self.word_to_token[token_text] = token
		self.token_to_word[token] = token_text

	def remove_token(self, token):
		derivative_tokens = self.inv_index[token]
		# for removed token's expansion, add each derivative token to their inv_index
		token_expansion = self.token_expansion[token]
		for t in token_expansion:
			self.inv_index[t].remove(token)
			self.inv_index[t].extend(derivative_tokens)
		# for each derivative token, update its expansion in token_expansion
		for dt in derivative_tokens:
			new_expansion = []
			for t in self.token_expansion[dt]:
				if t == token:
					new_expansion.extend(token_expansion)
				else:
					new_expansion.append(t)
			self.token_expansion[dt] = new_expansion

		del self.word_to_token[self.token_to_word[token]]
		del self.token_to_word[token]
		del self.inv_index[token]
		del self.token_expansion[token]

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

	def decompress_once(self, encoded_text: list[int], token: int) -> list[int]:
		# takes an encoded text and a token, and replaces all its occurences with its expansion
		ret = []
		expansion = self.token_expansion[token]
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
my_bpe = GreedyTokenizer(data, 600, 60)