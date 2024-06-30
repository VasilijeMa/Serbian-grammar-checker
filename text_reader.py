import string
from preprocessor import preprocess_sentence

DATA_SIZE = 4
TEXTS_FILE_NAME = "texts.txt"
VOCAB_FILE_NAME = "dict.txt"
LETTERS = string.ascii_letters + " -čćđšžČĆĐŠŽ1234567890"

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