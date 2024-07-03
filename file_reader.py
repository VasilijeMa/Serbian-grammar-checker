import string
from preprocessor import preprocess_sentence
import pandas as pd
import torch

DATA_SIZE = 1600
LETTERS = string.ascii_letters + " -čćđšžČĆĐŠŽ1234567890"

TRAIN_FILE_NAME = "data/texts_per_line.txt"
TEST_FILE_NAME = "data/test_sentences_84.csv"
VOCAB_FILE_NAME = "data/dict.txt"

def get_sentences():
    sentences = []
    with open(TRAIN_FILE_NAME, 'r', encoding='utf-8') as file:
        i = 0
        for line in file:
            i+=1
            sentences.append(preprocess_sentence(line.strip(), LETTERS))
            if i==DATA_SIZE: break
    return sentences

def get_vocab():
    vocab = []
    with open(VOCAB_FILE_NAME, 'r', encoding='utf-8') as file:
        for line in file:
            vocab.append(line.strip())
    return vocab

def get_test_data():
    df = pd.read_csv(TEST_FILE_NAME)
    return df['test'], df['correct']

def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model