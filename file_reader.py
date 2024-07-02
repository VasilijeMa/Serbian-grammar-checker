import string
from preprocessor import preprocess_sentence
import pandas as pd
import json

DATA_SIZE = 200
LETTERS = string.ascii_letters + " -čćđšžČĆĐŠŽ1234567890"

TEXTS_FILE_NAME = "texts.txt"
TEST_FILE_NAME = "test_sentences.csv"
VOCAB_FILE_NAME = "dict.txt"

def get_sentences():
    sentences = []
    with open(TEXTS_FILE_NAME, 'r', encoding='utf-8') as file:
        i = 0
        for line in file:
            i+=1
            lines = line.split("\t")
            sentences.append(preprocess_sentence(lines[1].strip(), LETTERS))
            sentences.append(preprocess_sentence(lines[2].strip(), LETTERS))
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