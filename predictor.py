import torch
import torch.nn.functional as F
from Levenshtein import distance as levenshtein_distance

PROB_THRESHOLD = 0.8

def encode_word(word):
    sum = 0.0
    power = 2.0
    for char in reversed(word):
        sum+=(ord(char)**power)
        power*=0.5
    return sum

def predict_between(model, word_before, word_after):
    model.eval()
    
    indices_before = [word_before]
    indices_after = [word_after]
    
    input_before = torch.tensor([indices_before], dtype=torch.long).unsqueeze(1)
    input_after = torch.tensor([indices_after], dtype=torch.long).unsqueeze(1)
    
    with torch.no_grad():
        output = model(input_before, input_after)
        probabilities = F.softmax(output, dim=1)

        max_prob, predicted_value = torch.max(probabilities, 1)

        max_prob = max_prob.item()
        predicted_value = predicted_value.item()
    
    return max_prob, predicted_value

def insert_word(model, word_pair, inverted_vocab):
    #TODO
    pass

def insert_into_sentence(model, original_sentence, inverted_vocab):
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

def replace_word(model, word_pair, inverted_vocab, original_word):
    #TODO
    pass

def replace_in_sentence(model, original_sentence, inverted_vocab):
    n = 0
    prob_sum = 0.0
    word_pairs = []
    for i in range(len(original_sentence) - 2):
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
    new_sentence.append(original_sentence[-1])
    return prob_sum, n, new_sentence