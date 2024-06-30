import string
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *

TEXTS_FILE_NAME = "texts.txt"
VOCAB_FILE_NAME = "dict.txt"
LETTERS = string.ascii_letters + " -čćđšžČĆĐŠŽ1234567890"

VOCAB_SIZE = 150000
EMBEDDING_DIM = 64
HIDDEN_DIM = 128

PROB_THRESHOLD = 0.8

# Edit these parameters to optimize performance
DATA_SIZE = 4
NUM_EPOCHS = 20
BUCKET_RANGE = 10

def get_solutions(model, word_pair, inverted_vocab):
    max_prob, predicted_value = predict_between(model, encode_word(word_pair[0]), encode_word(word_pair[1]))
    print(f"Predicted {predicted_value} with probability {max_prob}")
    solutions = []
    for key in inverted_vocab.keys():
        if key - predicted_value > BUCKET_RANGE: break
        if abs(key - predicted_value) > BUCKET_RANGE: continue
        solutions.append(inverted_vocab[key])
    return solutions

def insert_word():
    #TODO
    pass

def replace_word():
    #TODO
    pass

def insert_words(model, original_sentence, inverted_vocab)
    n = 0
    prob_sum = 0.0
    word_pairs = []
    for i in range(len(original_sentence) - 1):
        word_pairs.append([original_sentence[i], original_sentence[i+1]])

    new_sentence = []
    for word_pair in word_pairs:
        new_sentence.append(word_pair[0])
        prob, new_word = insert_word(model, word_pair, inverted_vocab)

        if prob >= PROB_THRESHOLD:
            prob_sum += prob
            n += 1
            new_sentence.append(new_word)
    new_sentence.append(original_sentence[-1])
    return prob_sum, n, new_sentence

def replace_words(model, original_sentence, inverted_vocab):
    n = 0
    prob_sum = 0.0
    word_pairs = []
    for i in range(len(new_sentence_insert) - 2):
        word_pairs.append([original_sentence[i], original_sentence[i+2]])

    new_sentence = []
    for i, word_pair in enumerate(word_pairs):
        original_word = original_sentence[i+1]
        new_sentence.append(word_pair[0])
        prob, new_word = replace_word(model, word_pair, inverted_vocab, original_word)

        if prob >= PROB_THRESHOLD:
            prob_sum += prob
            n += 1
            new_sentence.append(new_word)
        else: new_sentence.append(original_word)
    new_sentence.append(spellchecked_sentence[-1])
    return prob_sum, n, new_sentence

if __name__ == "__main__":
    sentences = []
    spelling_vocab = []

    with open(TEXTS_FILE_NAME, 'r', encoding='utf-8') as file:
        i = 0
        for line in file:
            i+=1
            lines = line.split("\t")
            sentences.append(preprocess_sentence(lines[1].strip(), LETTERS))
            sentences.append(preprocess_sentence(lines[2].strip(), LETTERS))
            if i==DATA_SIZE: break

    with open(VOCAB_FILE_NAME, 'r', encoding='utf-8') as file:
        for line in file:
            spelling_vocab.append(line.strip())

    unique_words = set(word for sentence in sentences for word in sentence)
    vocab = {word: encode_word(word) for word in unique_words}

    inverted_vocab = {}
    for word, value in vocab.items():
        if value not in inverted_vocab:
            inverted_vocab[value] = []
        inverted_vocab[value].append(word)
    inverted_vocab = dict(sorted(inverted_vocab.items()))

    sequences = [[vocab[word] for word in sentence] for sentence in sentences]

    X_before, X_after, y = create_io_pairs(sequences)

    X_before = torch.tensor(X_before, dtype=torch.long).unsqueeze(1)
    X_after = torch.tensor(X_after, dtype=torch.long).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.long)

    dataset = WordDataset(X_before, X_after, y)
    dataloader = DataLoader(dataset, batch_size=DATA_SIZE, shuffle=True)

    model = WordBetweenModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(NUM_EPOCHS):
        print(f"Starting epoch {epoch+1}")
        i = 1
        for input_before, input_after, target in dataloader:
            input_before = input_before.unsqueeze(1)
            input_after = input_after.unsqueeze(1)

            outputs = model(input_before, input_after)
            loss = criterion(outputs, target)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

    test_sentences = ['Minisđar poslo,va Slovačka Miroslav',
                      'ranjen snajpera u glava u blizini prelaza',
                      'Stanov za Rome']

    for sentence in test_sentences:
        clean_sentence = preprocess_sentence(sentence, LETTERS)
        print(f"Cleaned sentence: {clean_sentence}")
        corrected_versions = {}

        for spellchecked_sentence in spellcheck_sentence(clean_sentence, spelling_vocab):
            print(f"Spellchecked sentence: {spellchecked_sentence}")
            n1 = 0
            prob_sum_1 = 0.0
            n2 = 0
            prob_sum_2 = 0.0

            # Replace after insert

            print("-------- insert 1 ----------")

            prob, n, new_sentence_insert = insert_words(model, spellchecked_sentence, inverted_vocab)
            prob_sum_1 += prob
            n1 += n
                
            print("-------- replace 1 ---------")
            
            prob, n, sentence_1 = replace_words(model, new_sentence_insert, inverted_vocab)
            prob_sum_1 += prob
            n1 += n

            # Insert after replace
            
            print("-------- replace 2 ----------")

            prob, n, new_sentence_replace = replace_words(model, spellchecked_sentence, inverted_vocab)
            prob_sum_2 += prob
            n2 += n
            
            print("-------- insert 2 ---------")

            prob, n, sentence_2 = insert_words(model, new_sentence_replace, inverted_vocab)
            prob_sum_2 += prob
            n2 += n

            prob_avg_1 = prob_sum_1/n1
            prob_avg_2 = prob_sum_2/n2

            if prob_avg_1 > prob_avg_2:
                corrected_versions[prob_avg_1] = sentence_1
            else:
                corrected_versions[prob_avg_2] = sentence_2
        
        max_prob = max(corrected_versions.keys())
        print(f"Best version with probability {max_prob} is {' '.join(corrected_versions[max_prob])}")