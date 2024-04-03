# Simple Byte Pair Encoding implementation that directly encodes characters 
# to tokens and vice-versa, without going into the bytestream details.
# the 

import torch
import torch.nn.functional as F
import numpy as np
import math
import collections

class BytePairEncoding:
	def __init__(self, text: str, max_vocab_size: int, token_min_freq: int, skip_chars = [' ', ',', '.', '\n', '\t']):
		# generate encoding and decoding tables from text
		self.chars = sorted(list(set(text)))
		self.vocab = { ch:i for i,ch in enumerate(self.chars) }
		self.inv_vocab = { i:ch for i,ch in enumerate(self.chars) }

		max_vocab_size = min(max_vocab_size, math.sqrt(len(text)))
		encoded_text = [self.vocab[ch] for ch in text]
		self.tokens = {}
		self.index = {}
		self.next_token = len(self.vocab)
		self.skip_tokens = set([self.vocab.get(x, -1) for x in skip_chars])
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
			# Add most frequent bigram to tokens and index
			self.tokens[self.next_token] = top_bigram
			self.index[top_bigram] = self.next_token
			# unroll token for faster decoding
			token_text = self.unroll(self.next_token)
			self.vocab[token_text] = self.next_token
			self.inv_vocab[self.next_token] = token_text
			self.next_token += 1
			# update text
			encoded_text = self.compress_once(encoded_text)
		self.print_codebook()

	def print_codebook(self):
		ans = []
		for token in self.tokens:
			ans.append(self.unroll(token))
		print(ans)

	def unroll(self, token: int) -> str:
		if token not in self.tokens and token not in self.inv_vocab:
			return "===== UNKNOWN ====="
		if token in self.inv_vocab:
			return self.inv_vocab[token]
		token_list = []
		expanded = [token]
		while len(expanded) > len(token_list):
			token_list = expanded
			expanded = []
			for x in token_list:
				if x in self.tokens:
					expanded.extend(self.tokens[x])
				else:
					expanded.append(x)
		ret = "".join([self.inv_vocab[x] for x in expanded])
		return ret


	def get_bigrams(self, text: list[int]):
		ret = []
		for bigram in zip(text, text[1:]):
			if (bigram[0] not in self.skip_tokens) and (bigram[1] not in self.skip_tokens):
				ret.append(bigram)
		return ret

	def compress_once(self, encoded_text: list[int]) -> list[int]:
		# takes an encoded text, and replaces all directly tokenisable
		# bigram pairs, according to index
		ret = []
		i = 0
		while i < len(encoded_text)-1:
			bigram = (encoded_text[i], encoded_text[i+1])
			if bigram in self.index:
				ret.append(self.index[bigram])
				i += 2
			else:
				ret.append(encoded_text[i])
				i += 1
		if i < len(encoded_text):
			ret.append(encoded_text[i])
			i += 1
		return ret

	def encode(self, text: str) -> list[int]:
		encoded = []
		max_token_len = len(max(self.vocab, key=len))
		i = 0
		while i < len(text):
			for j in range(max_token_len, 0, -1):
				if i + j > len(text):
					continue
				maybe_token = text[i: i+j]
				if maybe_token in self.vocab:
					encoded.append(self.vocab[maybe_token])
					break
			i += j
		return encoded

	def decode(self, tokens: list[int]) -> str:
		ans = []
		for i in tokens:
			ans.append(self.inv_vocab[i] if i in self.inv_vocab else "<UNKOWN>")
		return "".join(ans)

# data I/O
data = open('wot1.txt', 'r').read()
my_bpe = BytePairEncoding(data, 200, 3)
wiki = ["The Wheel of Time is a series of high fantasy novels by American author Robert Jordan, with Brandon Sanderson as a co-author for the final three installments.",
	"Originally planned as a six-book series with the publication of The Eye of the World in 1990, The Wheel of Time came to span 14 volumes, in addition to a prequel novel and three companion books.",
	"Jordan died in 2007 while working on what was planned to be the twelfth and final volume in the series.",
	"He prepared extensive notes, which enabled fellow fantasy author Sanderson to complete the final book, which grew into three volumes: The Gathering Storm (2009), Towers of Midnight (2010), and A Memory of Light (2013).",
	"The series draws on numerous elements of both European and Asian mythology, most notably the cyclical nature of time found in Buddhism and Hinduism; the metaphysical concepts of balance, duality, and a respect for nature found in Taoism; and the dualistic concepts of God and Satan.",
	"The Wheel of Time is notable for its length, detailed imaginary world, magic system, and its large cast of characters.",
	"The eighth through fourteenth books each reached number one on the New York Times Best Seller list. After its completion, the series was nominated for a Hugo Award. As of 2021, the series has sold over 90 million copies worldwide, making it one of the best-selling epic fantasy series since The Lord of the Rings. Its popularity has spawned comic book adaptations, a collectible card game, a video game, a roleplaying game, and a soundtrack album. A television series adaptation produced by Sony Pictures and Amazon Studios premiered in 2021."]
print("original length: ", sum([len(x) for x in wiki]))
encoded_wiki = [my_bpe.encode(x) for x in wiki]
print(encoded_wiki)
print("encoded length: ", sum([len(x) for x in encoded_wiki]))
decoded = [my_bpe.decode(x) for x in encoded_wiki]
print(decoded)
print([x == y for x, y in zip(decoded, wiki)])