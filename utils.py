import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from Levenshtein import distance as levenshtein_distance

class Tokenizer:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        self.mask_token = '<MASK>'

    def fit(self, sentences):
        unique_words = set(word for sentence in sentences for word in sentence.split())
        self.vocab_size = len(unique_words) + 3  # Add 3 for <PAD>, <EOS>, and <MASK>
        self.word2idx = {word: idx + 1 for idx, word in enumerate(unique_words)}
        self.word2idx['<PAD>'] = 0
        self.word2idx['<EOS>'] = self.vocab_size - 2
        self.word2idx[self.mask_token] = self.vocab_size - 1
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def encode(self, sentence):
        return [self.word2idx[word] for word in sentence.split()] + [self.word2idx['<EOS>']]

    def decode(self, indices):
        return ' '.join(self.idx2word[idx] for idx in indices if idx not in (0, self.word2idx['<EOS>']))

    def convert_tokens_to_ids(self, tokens):
        return [self.word2idx.get(token, self.word2idx['<PAD>']) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.idx2word[id] for id in ids]

    def convert_tokens_to_string(self, tokens):
        return ' '.join(tokens)

    def insert(self, tokens, position, token):
        return tokens[:position] + [token] + tokens[position:]



