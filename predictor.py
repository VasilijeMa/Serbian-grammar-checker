import torch
import torch.nn.functional as F
from Levenshtein import distance as levenshtein_distance

PROB_THRESHOLD = 0.4
BUCKET_RANGE = 10

def get_magic_numbers(model, sentences):
    probs = []
    distances = []
    hits = 0
    for sentence in sentences:
        print(sentences.index(sentence))
        for i in range(1,len(sentence)-1):

            prob, predicted_word = predict_between(model, encode_word(sentence[i-1]), encode_word(sentence[i+1]))
            distances.append(abs(encode_word(sentence[i]) - predicted_word))

            print(f"Original: {encode_word(sentence[i])}")
            print(f"Middle: {predicted_word}")
            probs.append(prob)
    bucket_range = sum(distances) / len(distances)
    prob_threshold = 0.0
    hits = 0
    for i in range(len(distances)):
        if distances[i] < bucket_range:
            prob_threshold += probs[i]
            hits += 1
    return prob_threshold/hits, bucket_range

def encode_word(word):
    sum = 0.0
    power = 2.0
    for char in reversed(word):
        sum+=(ord(char)**power)
        power*=0.5
    return sum/1000

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

    max_prob, predicted_value = predict_between(model, encode_word(word_pair[0]), encode_word(word_pair[1]))
    predicted_word = inverted_vocab[min(inverted_vocab.keys(), key=lambda k: abs(k - predicted_value))]
    return max_prob, predicted_word

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


def replace_word(model, word_pair, inverted_vocab, original_word, bucket_range):
    max_prob, predicted_value = predict_between(model, encode_word(word_pair[0]), encode_word(word_pair[1]))

    solutions = []
    for key in inverted_vocab.keys():
        if key - predicted_value > bucket_range: break
        if abs(key - predicted_value) > bucket_range: continue
        solutions.append(inverted_vocab[key])

    if len(solutions) == 0: return max_prob, original_word

    predicted_word = min(solutions, key=lambda s: levenshtein_distance(s, original_word))
    return max_prob, predicted_word

def replace_in_sentence(model, original_sentence, inverted_vocab, prob_treshold, bucket_range):
    n = 0
    prob_sum = 0.0
    word_pairs = []
    for i in range(len(original_sentence) - 2):
        word_pairs.append([original_sentence[i], original_sentence[i+2]])

    skip = False
    new_sentence = []
    for i, word_pair in enumerate(word_pairs):
        original_word = original_sentence[i + 1]
        if skip:
            if i == len(word_pairs) - 1:
                new_sentence.append(original_word)

            skip = False
            continue

        new_sentence.append(word_pair[0])
        prob, new_word = replace_word(model, word_pair, inverted_vocab, original_word, bucket_range)

        if new_word == original_word:
            if i == len(word_pairs)-1:
                new_sentence.append(original_word)

            continue
        if prob >= prob_treshold:
            skip = True
            prob_sum += prob
            n += 1
            new_sentence.append(new_word)
        elif i == len(word_pairs)-1:
            new_sentence.append(original_word)

    new_sentence.append(original_sentence[-1])
    if n > 0 : prob_sum /= n
    return prob_sum, new_sentence


def calculate_accuracy(predicted_sentence, correct_sentence):

    distance = levenshtein_distance(predicted_sentence, correct_sentence)
    max_length = max(len(predicted_sentence), len(correct_sentence))
    normalized_distance = distance/max_length

    accuracy = (1 - normalized_distance) * 100
    return accuracy