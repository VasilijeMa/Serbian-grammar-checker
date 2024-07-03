def preprocess_sentence(sentence, letters):
    return ''.join(letter for letter in sentence if letter in letters).split()

def create_io_pairs(sequences):
    X_before, X_after, y = [], [], []
    for sequence in sequences:
        if len(sequence) > 2:
            for i in range(1, len(sequence)-1):
                X_before.append(sequence[i-1])
                X_after.append(sequence[i+1])
                y.append(sequence[i])
    return X_before, X_after, y

def invert(vocab):
    inverted_vocab = {}
    for word, value in vocab.items():
        inverted_vocab[value] = word

    return dict(sorted(inverted_vocab.items()))
