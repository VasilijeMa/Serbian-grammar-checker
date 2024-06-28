import string
from utils import *
from collections import Counter

TEXTS_FILE_NAME = "texts.txt"
VOCAB_FILE_NAME = "dict.txt"
LETTERS = string.ascii_letters + " -čćđšžČĆĐŠŽ"

VOCAB_SIZE = 150000
EMBEDDING_DIM = 64
HIDDEN_DIM = 128

# Edit these parameters to optimize performance
DATA_SIZE = 100
BATCH_SIZE = 32
NUM_EPOCHS = 20
BUCKET_RANGE = 10

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

    tokenized_sentences = [sentence.split() for sentence in sentences]
    unique_words = set(word for sentence in tokenized_sentences for word in sentence)
    vocab = {word: encode_word(word) for word in unique_words}

    inverted_vocab = {}
    for word, value in vocab.items():
        if value not in inverted_vocab:
            inverted_vocab[value] = []
        inverted_vocab[value].append(word)
    inverted_vocab = dict(sorted(inverted_vocab.items()))

    sequences = [[vocab[word] for word in sentence] for sentence in tokenized_sentences]

    X_before, X_after, y = create_io_pairs(sequences)

    X_before = torch.tensor(X_before, dtype=torch.long).unsqueeze(1)
    X_after = torch.tensor(X_after, dtype=torch.long).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.long)

    dataset = WordDataset(X_before, X_after, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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

    predicted_value = predict_between(model, encode_word("Pripadnici"), encode_word("agencija"))
    solutions = []
    for key in inverted_vocab.keys():
        if key - predicted_value > BUCKET_RANGE: break
        if abs(key - predicted_value) > BUCKET_RANGE: continue
        solutions.append(inverted_vocab[key])
    print(solutions)
    


