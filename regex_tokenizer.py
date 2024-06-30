# A greedy tokenizer inspired by BytePairEncoding that directly encodes characters 
# to tokens and vice-versa, without going into the bytestream details (unlike tiktoken).
# This one chucks out infrequent tokens once they've been used up by larger ones.
# For example, if the tokenizer learnt a token for "tok", but later learnt another
# token for "token", such that "tok" is no longer a frequent token by itself,
# it will free up the token id for "tok" and add something else to the vocab instead.
# Also, while selecting the next token to add to the vocab, it slightly prefers smaller
# tokens over larger ones.

# TODO: [perf] use trie for faster/cleaner encoding.
# TODO: [perf] keep track of bigram freqs while swapping tokens too.

import math
import collections
import pickle
import os
from datetime import datetime
import regex as re
import itertools

def get_bigrams(encoded_text: list[list[int]]) -> list[(int, int)]:
	ret = []
	for word in encoded_text:
		if len(word) > 1:
			for bigram in zip(word, word[1:]):
				ret.append(bigram)
	return ret

def get_top_bigram(bigram_freqs, bigram_lengths = None):
	if bigram_lengths is not None:
		top_bigram = max(bigram_freqs,
			key=lambda x: bigram_freqs[x] / math.sqrt(bigram_lengths[x]))
	else:
		top_bigram = max(bigram_freqs, key=bigram_freqs.get)
	return top_bigram, bigram_freqs[top_bigram]

def update_bigram_freqs(new_token: int, bigram_freqs, token_indices: list[(int, int)], encoded_text: list[list[int]], encoded_bigram):
	del bigram_freqs[encoded_bigram]
	for i, j in token_indices:
		if encoded_text[i][j] != new_token:
			raise Exception("ERROR: new token not found at specified index.")
		phrase = encoded_text[i]
		if j + 1 < len(phrase):
			next_token = phrase[j+1]
			if (new_token, next_token) not in bigram_freqs:
				bigram_freqs[(new_token, next_token)] = 0
			bigram_freqs[(new_token, next_token)] += 1
			if next_token != new_token:
				bigram_freqs[(encoded_bigram[1], next_token)] -= 1
			else:
				bigram_freqs[(encoded_bigram[1], encoded_bigram[0])] -= 1
		if j > 0:
			prev_token = phrase[j-1] 
			if prev_token != new_token:
				if (prev_token, new_token) not in bigram_freqs:
					bigram_freqs[(prev_token, new_token)] = 0
				bigram_freqs[(prev_token, new_token)] += 1
				bigram_freqs[(prev_token, encoded_bigram[0])] -= 1
	return bigram_freqs


class RegexTokenizer:
	def __init__(self, text: str, max_vocab_size: int = 1000,
		token_min_freq: int = 50,
		skip_chars = [' ', ',', '.', '\n', '\t', '"'],
		split_regex_pattern = \
			r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""):
		# generate encoding and decoding tables from text
		self.chars = sorted(list(set(text)))
		print(f'{len(self.chars)} Raw characters:', self.chars)
		self.word_to_token = { ch:i for i,ch in enumerate(self.chars) }
		self.token_to_word = { i:ch for i,ch in enumerate(self.chars) }

		self.regex_pattern = split_regex_pattern
		self.compiled_pattern = re.compile(self.regex_pattern)

		max_vocab_size = min(max_vocab_size, math.sqrt(len(text)))
		text_chunks = re.findall(self.compiled_pattern, text)
		print("split text chunks:", text_chunks[:10])
		encoded_text = [[self.word_to_token[ch] for ch in phrase] for phrase in text_chunks]
		self.token_expansion = {} 	# token to sequence of constituent tokens
		self.inv_index = {} 		# token to list of tokens that rely on this token
		self.next_token = len(self.word_to_token)
		# self.skip_tokens = set([self.word_to_token.get(x, -1) for x in skip_chars])

		encoded_text = self.build_vocab(max_vocab_size, encoded_text, token_min_freq)
		self.vocab_swapping(encoded_text, token_min_freq)
		self.print_codebook()

	def vocab_size(self):
		return len(self.word_to_token)

	def get_bigram_lengths(self, bigrams):
		lengths = {}
		for bigram in bigrams:
			lengths[bigram] = len(self.token_to_word[bigram[0]]) + len(self.token_to_word[bigram[1]])
		return lengths

	def build_vocab(self, max_vocab_size: int, encoded_text: list[list[int]], token_min_freq):
		bigram_freqs = collections.Counter(get_bigrams(encoded_text))
		print("Vocab Size:", self.next_token)
		print("Text Length:", sum([len(phrase) for phrase in encoded_text]))
		while self.next_token < max_vocab_size:
			if self.next_token % 10 == 0:
				print("Vocab Size:", self.next_token)
				print("Text Length:", sum([len(phrase) for phrase in encoded_text]))
			bigram_lengths = self.get_bigram_lengths(bigram_freqs)
			top_bigram, top_bigram_freq = get_top_bigram(bigram_freqs, bigram_lengths)
			if top_bigram_freq < token_min_freq:
				break
			self.add_token(self.next_token, top_bigram)
			encoded_text, encoded_indices = self.compress_once(encoded_text, top_bigram, self.next_token)
			bigram_freqs = update_bigram_freqs(self.next_token, bigram_freqs, encoded_indices, encoded_text, top_bigram)
			self.next_token += 1
		return encoded_text

	def vocab_swapping(self, encoded_text: list[list[int]], token_min_freq):
		# remove infrequent tokens to free up space for newer compound tokens.
		swap_count = 0
		while True:
			bigram_freqs = collections.Counter(get_bigrams(encoded_text))
			# don't look at bigram length while swapping.
			top_bigram, top_bigram_freq = get_top_bigram(bigram_freqs)
			if top_bigram_freq < token_min_freq:
				print("No frequent bigram found:", self.token_to_word[top_bigram[0]], self.token_to_word[top_bigram[1]], "freq:", top_bigram_freq)
				break
			unigram_freqs = collections.Counter(itertools.chain.from_iterable(encoded_text))
			popped_token = min(self.token_expansion, \
				key=lambda x: unigram_freqs.get(x, 0) if x >= len(self.chars) else float('inf'))
			if popped_token < len(self.chars):
				print("No infrequent token found for swapping.")
				break
			if top_bigram_freq < 2 * unigram_freqs[popped_token] and unigram_freqs[popped_token] >= token_min_freq:
				print("incumbent:", self.token_to_word[popped_token], "freq:", unigram_freqs[popped_token], \
					"challenger freq:", top_bigram_freq)
				break
			print("Replacing token:", self.token_to_word[popped_token], "with freq:", unigram_freqs[popped_token])
			encoded_text = self.decompress_once(encoded_text, popped_token)
			self.remove_token(popped_token)
			self.add_token(popped_token, top_bigram)
			print("Replaced with:", self.token_to_word[popped_token])
			swap_count += 1
			encoded_text, _ = self.compress_once(encoded_text, top_bigram, popped_token)
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
		derivative_tokens = self.inv_index.get(token, [])
		# for removed token's expansion, add each derivative token to their inv_index
		token_expansion = self.token_expansion[token]
		for t in token_expansion:
			self.inv_index[t].remove(token)
			self.inv_index[t].extend(derivative_tokens)
		# for each derivative token, update its expansion
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
		if token in self.inv_index:
			del self.inv_index[token]
		del self.token_expansion[token]

	def print_codebook(self):
		ans = []
		ans.extend(self.chars)
		for token in self.token_expansion:
			ans.append(self.unroll(token))
		print(sorted(ans))

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

	def decompress_once(self, encoded_text: list[list[int]], token: int) -> list[int]:
		# takes an encoded text and a token, and replaces all its occurences with its expansion
		ret = []
		expansion = self.token_expansion[token]
		for compressed in encoded_text:
			decompressed = []
			for x in compressed:
				if x == token:
					decompressed.extend(expansion)
				else:
					decompressed.append(x)
			ret.append(decompressed)
		return ret

	def compress_once(self, encoded_text: list[list[int]], bigram: tuple[int], token: int) -> (list[list[int]], list[tuple[int]]):
		# takes encoded texts and a bigram, and replaces all its occurences with token
		compressed_text = []
		indices = []
		for i, encoded_word in enumerate(encoded_text):
			compressed_word = []
			j = 0
			while j < len(encoded_word)-1:
				curr_bigram = (encoded_word[j], encoded_word[j+1])
				if curr_bigram == bigram:
					indices.append((i, len(compressed_word)))
					compressed_word.append(token)
					j += 2
				else:
					compressed_word.append(encoded_word[j])
					j += 1
			if j < len(encoded_word):
				compressed_word.append(encoded_word[j])
			compressed_text.append(compressed_word)
		return compressed_text, indices

	def encode(self, text: str) -> list[int]:
		encoded = []
		max_token_len = len(max(self.word_to_token, key=len))
		i = 0
		while i < len(text):
			for j in range(max_token_len, 0, -1):
				if i + j > len(text):
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

	def save(self) -> str:
		current_date = datetime.now().strftime('%Y%m%d')
		filename = f'regex_tokenizer_{len(self.token_to_word)}_{current_date}.pkl'
		# Save the object to a file
		print("Saving object to file...")
		with open(filename, "wb") as file:
		    pickle.dump(self, file)

		print(f"Object saved to {filename}")
		return filename

	def load(filename: str):
		print("\nReading tokenizer object from file...")
		with open(filename, "rb") as file:
		    tokenizer = pickle.load(file)
		return tokenizer

# data = open('wot1.txt', 'r').read()
# data += open('wot2.txt', 'r').read()
# data += open('wot3.txt', 'r').read()
# my_bpe = RegexTokenizer(data, 200, 250)
# wiki = ["The Wheel of Time is a series of high fantasy novels by American author Robert Jordan, with Brandon Sanderson as a co-author for the final three installments.",
# 	"Originally planned as a six-book series with the publication of The Eye of the World in 1990, The Wheel of Time came to span 14 volumes, in addition to a prequel novel and three companion books.",
# 	"Jordan died in 2007 while working on what was planned to be the twelfth and final volume in the series.",
# 	"He prepared extensive notes, which enabled fellow fantasy author Sanderson to complete the final book, which grew into three volumes: The Gathering Storm (2009), Towers of Midnight (2010), and A Memory of Light (2013).",
# 	"The series draws on numerous elements of both European and Asian mythology, most notably the cyclical nature of time found in Buddhism and Hinduism; the metaphysical concepts of balance, duality, and a respect for nature found in Taoism; and the dualistic concepts of God and Satan.",
# 	"The Wheel of Time is notable for its length, detailed imaginary world, magic system, and its large cast of characters.",
# 	"The eighth through fourteenth books each reached number one on the New York Times Best Seller list. After its completion, the series was nominated for a Hugo Award. As of 2021, the series has sold over 90 million copies worldwide, making it one of the best-selling epic fantasy series since The Lord of the Rings. Its popularity has spawned comic book adaptations, a collectible card game, a video game, a roleplaying game, and a soundtrack album. A television series adaptation produced by Sony Pictures and Amazon Studios premiered in 2021."]
# print("original length: ", sum([len(x) for x in wiki]))
# encoded_wiki = [my_bpe.encode(x) for x in wiki]
# print("encoded length: ", sum([len(x) for x in encoded_wiki]))
# decoded = [my_bpe.decode(x) for x in encoded_wiki]
# print(decoded)
# my_bpe.save()
# print([x == y for x, y in zip(decoded, wiki)])