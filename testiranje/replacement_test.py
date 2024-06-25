import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from Levenshtein import distance as levenshtein_distance
import numpy as np

# Sample data: list of correct sentences
sentences = [
    "the cat sat on the mat",
    "the dog barked loudly",
    "a quick brown fox jumps over the lazy dog",
    "the rain in spain stays mainly in the plain",
    "ide gas ali nikako da ode"
]


# Tokenizer
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


# Prepare data
tokenizer = Tokenizer()
tokenizer.fit(sentences)
encoded_sentences = [torch.tensor(tokenizer.encode(sentence)) for sentence in sentences]
padded_sentences = pad_sequence(encoded_sentences, batch_first=True, padding_value=tokenizer.word2idx['<PAD>'])


class SentenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][:-1], self.data[idx][1:]


dataset = SentenceDataset(padded_sentences)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# Model
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output


# Hyperparameters
embedding_dim = 64
hidden_dim = 128
vocab_size = tokenizer.vocab_size

model = LanguageModel(vocab_size, embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 300

for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

def correct_sentence_replace(model, tokenizer, input_sentence, mask_position):
    model.eval()
    best_corrected_sentence = input_sentence
    best_accuracy = 0.0
    best_mask_position = -1

    with torch.no_grad():
        # Encode the input sentence
        tokens = input_sentence.split()
        removed_word = tokens.pop(mask_position)

        tokens.insert(mask_position, tokenizer.mask_token)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_seq = torch.tensor([input_ids])


        output = model(input_seq)
        masked_index = input_ids.index(tokenizer.word2idx[tokenizer.mask_token])

        masked_token_logits = output[0, masked_index-1, :]

        predicted_token_id = torch.argmax(masked_token_logits, dim=-1).item()
        #print(predicted_token_id)
        predicted_token = tokenizer.idx2word[predicted_token_id]

        tokens[masked_index] = predicted_token
        corrected_sentence = tokenizer.convert_tokens_to_string(tokens)




    return corrected_sentence

def correct_sentence_mask(model, tokenizer, input_sentence, sentences):
    model.eval()
    best_corrected_sentence = input_sentence
    best_accuracy = 0.0
    best_mask_position = -1

    # Try all possible mask positions
    for mask_position in range(len(input_sentence.split()) + 1):  # +1 to include the end of sentence position
        with torch.no_grad():
            # Encode the input sentence
            tokens = input_sentence.split()
            tokens.insert(mask_position, tokenizer.mask_token)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_seq = torch.tensor([input_ids])

            # Generate predictions for the masked token
            output = model(input_seq)
            masked_index = input_ids.index(tokenizer.word2idx[tokenizer.mask_token])
            masked_token_logits = output[0, masked_index, :]

            # Get the predicted token
            predicted_token_id = torch.argmax(masked_token_logits, dim=-1).item()
            predicted_token = tokenizer.idx2word[predicted_token_id]

            # Replace mask token with predicted token
            tokens[masked_index] = predicted_token

            # Decode the tokens back to a string
            corrected_sentence = tokenizer.convert_tokens_to_string(tokens)

            # Calculate accuracy


            for sentence in sentences:
                original_sentence = sentence

                accuracy = calculate_accuracy(corrected_sentence, original_sentence)

                # Update the best position if accuracy is higher
                if accuracy > best_accuracy:
                   best_accuracy = accuracy
                   best_corrected_sentence = corrected_sentence
                   best_mask_position = mask_position

    return best_corrected_sentence, best_mask_position


def calculate_accuracy(corrected_sentence, original_sentence):
    corrected_tokens = corrected_sentence.split()
    original_tokens = original_sentence.split()

    # Calculate Levenshtein distance
    distance = levenshtein_distance(corrected_tokens, original_tokens)

    # Calculate accuracy as a function of distance
    max_length = max(len(corrected_tokens), len(original_tokens))
    if max_length == 0:
        return 0.0
    accuracy = 1 - (distance / max_length)

    return accuracy
# Test prediction
# print(predict(model, tokenizer, ""))

#print(correct_sentence_mask(model, tokenizer, "a quick over dog", 2))
#corrected_sentence, best_mask_position = correct_sentence_mask(model, tokenizer, "ide gas ode", sentences)
corrected_sentence = correct_sentence_replace(model, tokenizer, "a quick brown fx jumps over the lazy dog", 3)
print(corrected_sentence)
#print(f"Corrected: {corrected_sentence} (Best mask position: {best_mask_position})")