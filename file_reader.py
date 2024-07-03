import string
from preprocessor import preprocess_sentence
import pandas as pd
import json

DATA_SIZE = 5
LETTERS = string.ascii_letters + " -čćđšžČĆĐŠŽ1234567890"

TEXTS_FILE_NAME = "Data/texts_per_line.txt"
TEST_FILE_NAME = "Data/test_sentences_170.csv"
VOCAB_FILE_NAME = "dict.txt"

VALID_FILE_NAME = "Data/validation.txt"
def get_validation_sentences():
    sentences = []
    with open(VALID_FILE_NAME, 'r', encoding='utf-8') as file:
        for line in file:
            sentences.append(preprocess_sentence(line.strip(), LETTERS))
    return sentences

def get_sentences():
    sentences = []
    with open(TEXTS_FILE_NAME, 'r', encoding='utf-8') as file:
        i = 0
        for line in file:
            i+=1
            sentences.append(preprocess_sentence(line.strip(), LETTERS))
            # For limiting training data size
          #  if i==DATA_SIZE: break
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