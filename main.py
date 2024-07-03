import os.path

import torch
from torch.utils.data import DataLoader

from classes import WordBetweenModel, WordDataset
from spellchecker import spellcheck_sentence
from predictor import replace_in_sentence, encode_word, calculate_accuracy
from preprocessor import *
from file_reader import *

import pandas as pd

from train import train

HIDDEN_DIM = 512
BATCH_SIZE = 128
NUM_EPOCHS = 100

if __name__ == "__main__":
    sentences = get_sentences()
    spelling_vocab = get_vocab()

    unique_words = set(word for sentence in sentences for word in sentence)
    inverted_vocab = {encode_word(word): word for word in unique_words}
    sequences = [[encode_word(word) for word in sentence] for sentence in sentences]

    X_before, X_after, y = create_io_pairs(sequences)
    X_before = torch.tensor(X_before, dtype=torch.float64)
    X_after = torch.tensor(X_after, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)

    dataloader = DataLoader(WordDataset(X_before, X_after, y), batch_size=BATCH_SIZE, shuffle=True)
    model = WordBetweenModel(HIDDEN_DIM)

    if not os.listdir('Models'):
        print("There are currently no models imported.")
        train(model, NUM_EPOCHS, dataloader, 'Models/model_{epoch}epochs.pth')
        
    # checkpoint_path = 'Models/' + os.listdir('Models')[0]
    # model = load_model(model, checkpoint_path)

    # test_sentences, correct_sentences = get_test_data()
    # results = []
    # others = []

    # print("Predicting...")

    # for sentence, correct_sentence in zip(test_sentences, correct_sentences):

    #     clean_sentence = preprocess_sentence(sentence, LETTERS)
    #     corrected_versions = {}

    #     for spellchecked_sentence in spellcheck_sentence(clean_sentence, spelling_vocab):
    #         prob, new_sentence = replace_in_sentence(model, spellchecked_sentence, inverted_vocab)
    #         corrected_versions[prob] = new_sentence

    #     max_prob = max(corrected_versions.keys())
    #     predicted_sentence = ' '.join(corrected_versions[max_prob])
    #     accuracy = calculate_accuracy(predicted_sentence, correct_sentence)

    #     others.append(sentence)
    #     others.append("*************************************")
    #     for key in corrected_versions.keys():
    #         if key == max_prob: continue
    #         others.append(f"Sentence: {' '.join(corrected_versions[key])}, probability: {key}")
    #     others.append("*************************************")

    #     result = {
    #         'test_sentence': sentence,
    #         'correct_sentence': correct_sentence,
    #         'predicted_sentence': predicted_sentence,
    #         'probability': max_prob,
    #         'accuracy': round(accuracy, 2)
    #     }

    #     results.append(result)

    # df_others = pd.DataFrame(others)
    # df_others.to_csv(f'Results/others_{NUM_EPOCHS}epochs_84.txt', index=False)

    # sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    # df_results = pd.DataFrame(sorted_results)
    # df_results.to_csv(f'Results/results_{NUM_EPOCHS}epochs_84.csv', index=False)

    # print("Successfully saved predicted results.")