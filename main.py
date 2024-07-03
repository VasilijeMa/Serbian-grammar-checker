import os.path
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from classes import WordBetweenModel, WordDataset
from spellchecker import spellcheck_sentence
from predictor import get_magic_numbers, replace_in_sentence, encode_word, calculate_accuracy
from preprocessor import *
from file_reader import *

import pandas as pd
from train import train

VOCAB_SIZE = 150000
EMBEDDING_DIM = 64
HIDDEN_DIM = 128

BATCH_SIZE = 256
NUM_EPOCHS = 20

PROB_THRESHOLD = 0.3
BUCKET_RANGE = 3

def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

if __name__ == "__main__":
    sentences = get_sentences()
    spelling_vocab = get_vocab()

    unique_words = set(word for sentence in sentences for word in sentence)
    vocab = {word: encode_word(word) for word in unique_words}
    inverted_vocab = invert(vocab)

    sequences = [[vocab[word] for word in sentence] for sentence in sentences]
    X_before, X_after, y = create_io_pairs(sequences)
    print(f"X before {X_before}")
    print(f"X after {X_after}")
    print(f"Y {y}")
    X_before = torch.tensor(X_before, dtype=torch.long).unsqueeze(1)
    X_after = torch.tensor(X_after, dtype=torch.long).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.long)

    dataloader = DataLoader(WordDataset(X_before, X_after, y), batch_size=BATCH_SIZE, shuffle=True)
    model = WordBetweenModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    max_probs = []
    accuracies = []
    if os.listdir('Models'):
        checkpoint_path = 'Models/' + os.listdir('Models')[0]
        model = load_model(model, checkpoint_path)

        test_sentences, correct_sentences = get_test_data()
        results = []
        others = []

        print(len(get_validation_sentences()))

        # Calculates optimal probability threshold and bucket range
       # prob_threshold, bucket_range = get_magic_numbers(model, get_validation_sentences())

        print("Predicting...")
        counter = 0
        length = len(test_sentences)
        for sentence, correct_sentence in zip(test_sentences, correct_sentences):

            if counter % 5 == 0:
                print(f"Progress: {counter * 100 / length}%")

            clean_sentence = preprocess_sentence(sentence, LETTERS)
            corrected_versions = {}

            for spellchecked_sentence in spellcheck_sentence(clean_sentence, spelling_vocab):
                prob, new_sentence = replace_in_sentence(model, spellchecked_sentence, inverted_vocab, PROB_THRESHOLD, BUCKET_RANGE)
                corrected_versions[prob] = new_sentence

            max_prob = max(corrected_versions.keys())
            predicted_sentence = ' '.join(corrected_versions[max_prob])
            accuracy = calculate_accuracy(predicted_sentence, correct_sentence)

            others.append(sentence)
            others.append("*************************************")
            for key in corrected_versions.keys():
                if key == max_prob: continue
                others.append(f"Sentence: {' '.join(corrected_versions[key])}, probability: {key}")
            others.append("*************************************")

            result = {
                'test_sentence': sentence,
                'correct_sentence': correct_sentence,
                'predicted_sentence': predicted_sentence,
                'probability': max_prob,
                'accuracy': round(accuracy, 2)
            }
            results.append(result)

            max_probs.append(max_prob)
            accuracies.append(round(accuracy, 2)/100)

            counter += 1

        df_others = pd.DataFrame(others)
        df_others.to_csv(f'Results/others_{NUM_EPOCHS}epochs.txt', index=False)

        sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
        df_results = pd.DataFrame(sorted_results)
        df_results.to_csv(f'Results/results_{NUM_EPOCHS}epochs.csv', index=False)

        print("Successfully saved predicted results.")

        plt.scatter(accuracies, max_probs)

        plt.show()
    else:
        print("There are currently no models imported.")
        train(model, criterion, optimizer, NUM_EPOCHS, dataloader, 'Models/model_{epoch}epochs.pth')

