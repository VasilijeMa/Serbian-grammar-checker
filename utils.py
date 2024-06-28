import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from Levenshtein import distance as levenshtein_distance
import itertools

class WordDataset(Dataset):
    def __init__(self, X_before, X_after, y):
        self.X_before = X_before
        self.X_after = X_after
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_before[idx], self.X_after[idx], self.y[idx]

class WordBetweenModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(WordBetweenModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x_before, x_after):
        embed_before = self.embedding(x_before)
        embed_after = self.embedding(x_after)
        
        combined = torch.cat((embed_before, embed_after), dim=2)
        combined = combined.squeeze(1)
        lstm_out, _ = self.lstm(combined)
        
        out = self.fc(lstm_out[:, -1, :])
        return out

def calculate_accuracy(corrected_sentence, original_sentence):
    corrected_tokens = corrected_sentence.split()
    original_tokens = original_sentence.split()

    distance = levenshtein_distance(corrected_tokens, original_tokens)

    max_length = max(len(corrected_tokens), len(original_tokens))
    if max_length == 0:
        return 0.0
    accuracy = 1 - (distance / max_length)

    return accuracy

def spellcheck_word(given_word, vocabulary):
    min_distance = 10*100
    nearest_words = []
    for word in vocabulary:
        if word.lower() == given_word.lower(): return [given_word]
        distance = levenshtein_distance(word, given_word)
        if distance > min_distance: continue
        if distance < min_distance:
            nearest_words.clear()
            
            min_distance = distance
        nearest_words.append(word)
    return nearest_words

def preprocess_sentence(sentence, letters):
    return ' '.join(''.join(letter for letter in sentence if letter in letters).split()) + " <EOS>"

def spellcheck_sentence(sentence, vocabulary):
    original_words = sentence.split()
    spellchecked_sentences = []
    words_to_replace = {}
    for i, word in enumerate(original_words):
        nearest_words = spellcheck_word(word, vocabulary)
        if nearest_words[0] == word: continue
        words_to_replace[i] = nearest_words
    
    pair_lists = [[(key, value) for value in values] for key, values in words_to_replace.items()]
    combinations = itertools.product(*pair_lists)
    for combination in combinations:
        spellchecked_sentence = original_words[:]
        for pair in combination:
            spellchecked_sentence[pair[0]] = pair[1]
        spellchecked_sentences.append(spellchecked_sentence)
    return spellchecked_sentences


def predict_between(model, context_before, context_after, vocab):
    model.eval()
    
    indices_before = [vocab[context_before]]
    indices_after = [vocab[context_after]]
    
    input_before = torch.tensor([indices_before], dtype=torch.long).unsqueeze(1)
    input_after = torch.tensor([indices_after], dtype=torch.long).unsqueeze(1)
    
    with torch.no_grad():
        output = model(input_before, input_after)
        _, predicted = torch.max(output, 1)
        predicted_word = predicted.item()
    
    inv_vocab = {i: word for word, i in vocab.items()}
    return ' '.join([context_before, inv_vocab[predicted_word], context_after])