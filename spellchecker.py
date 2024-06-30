from Levenshtein import distance as levenshtein_distance
import itertools

def spellcheck_word(given_word, vocabulary):
    min_distance = 10*100
    nearest_words = []
    for word in vocabulary:
        if word.lower() == given_word.lower(): return [given_word]
        distance = levenshtein_distance(word, given_word)
        if distance > min_distance: continue
        if distance < min_distance:
            nearest_words.clear()
            
            min_distance = distance
        nearest_words.append(word)
    return nearest_words

def spellcheck_sentence(sentence, vocabulary):
    spellchecked_sentences = []
    words_to_replace = {}
    for i, word in enumerate(sentence):
        if any(char.isdigit() for char in word): continue
        nearest_words = spellcheck_word(word, vocabulary)
        if nearest_words[0].lower() == word.lower(): continue
        words_to_replace[i] = nearest_words
    if len(words_to_replace) == 0: return [sentence]

    pair_lists = [[(key, value) for value in values] for key, values in words_to_replace.items()]
    combinations = itertools.product(*pair_lists)
    for combination in combinations:
        spellchecked_sentence = sentence[:]
        for pair in combination:
            spellchecked_sentence[pair[0]] = pair[1]
        spellchecked_sentences.append(spellchecked_sentence)
    return spellchecked_sentences