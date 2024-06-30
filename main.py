import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from classes import WordBetweenModel, WordDataset
from spellchecker import spellcheck_sentence
from predictor import insert_into_sentence, replace_in_sentence, encode_word
from preprocessor import *
from text_reader import *

VOCAB_SIZE = 150000
EMBEDDING_DIM = 64
HIDDEN_DIM = 128

BATCH_SIZE = 4
NUM_EPOCHS = 20

if __name__ == "__main__":
    sentences = get_sentences()
    spelling_vocab = get_vocab()

    unique_words = set(word for sentence in sentences for word in sentence)
    vocab = {word: encode_word(word) for word in unique_words}
    inverted_vocab = invert(vocab)

    sequences = [[vocab[word] for word in sentence] for sentence in sentences]
    X_before, X_after, y = create_io_pairs(sequences)
    X_before = torch.tensor(X_before, dtype=torch.long).unsqueeze(1)
    X_after = torch.tensor(X_after, dtype=torch.long).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.long)

    dataloader = DataLoader(WordDataset(X_before, X_after, y), batch_size=BATCH_SIZE, shuffle=True)
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
                       'Stanov za Rome',
                       'Tužilštvo pozval Miu Farou, Naomi Kembel njenog bvšeg']

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
            prob, n, new_sentence_insert = insert_into_sentence(model, spellchecked_sentence, inverted_vocab)
            prob_sum_1 += prob
            print(f"New insert: {new_sentence_insert}, prob: {prob}")
            n1 += n
                
            print("-------- replace 1 ---------")
            prob, n, sentence_1 = replace_in_sentence(model, new_sentence_insert, inverted_vocab)
            prob_sum_1 += prob
            print(f"New replace: {sentence_1}, prob: {prob}")
            n1 += n

            # Insert after replace
            
            print("-------- replace 2 ----------")
            prob, n, new_sentence_replace = replace_in_sentence(model, spellchecked_sentence, inverted_vocab)
            prob_sum_2 += prob
            print(f"New replace: {new_sentence_replace}, prob: {prob}")
            n2 += n
            
            print("-------- insert 2 ---------")
            prob, n, sentence_2 = insert_into_sentence(model, new_sentence_replace, inverted_vocab)
            prob_sum_2 += prob
            print(f"New insert: {sentence_2}, prob: {prob}")

            n2 += n

            prob_avg_1 = prob_sum_1
            prob_avg_2 = prob_sum_2
            # Calculate highest probability
            if n1 > 0:
                prob_avg_1 /= n1
            if n2 > 0:
                prob_avg_2 /= n2
            
            if prob_avg_1 > prob_avg_2:
                sentence_1[0] = sentence_1[0].capitalize()
                corrected_versions[prob_avg_1] = sentence_1
            else:
                sentence_2[0] = sentence_2[0].capitalize()
                corrected_versions[prob_avg_2] = sentence_2
        
        max_prob = max(corrected_versions.keys())
        print(f"Best version with probability {max_prob} is {' '.join(corrected_versions[max_prob])}\nOthers:")
        for key in corrected_versions.keys():
            if key == max_prob: continue
            print(f"Sentence: {' '.join(corrected_versions[key])}, probability: {key}")