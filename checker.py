import string
from utils import *
from collections import Counter

TEXTS_FILE_NAME = "texts.txt"
VOCAB_FILE_NAME = "dict.txt"
LETTERS = string.ascii_letters + " čćđšžČĆĐŠŽ"

EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_EPOCHS = 3

if __name__ == "__main__":
    sentences = []
    vocabulary = []

    with open(TEXTS_FILE_NAME, 'r', encoding='utf-8') as file:
        i = 0
        for line in file:
            i+=1
            lines = line.split("\t")
            sentences.append(preprocess_sentence(lines[1].strip(), LETTERS))
            sentences.append(preprocess_sentence(lines[2].strip(), LETTERS))
            if i==100: break

    with open(VOCAB_FILE_NAME, 'r', encoding='utf-8') as file:
        for line in file:
            vocabulary.append(line.strip())

    tokenized_sentences = [sentence.split() for sentence in sentences]
    unique_words = set(word for sentence in tokenized_sentences for word in sentence)

    vocab = {word: i+1 for i, word in enumerate(unique_words)}
    inv_vocab = {v: k for k, v in vocab.items()}

    sequences = [[vocab[word] for word in sentence] for sentence in tokenized_sentences]

    def create_io_pairs(sequences):
        X_before, X_after, y = [], [], []
        for sequence in sequences:
            if len(sequence) > 2:
                for i in range(1, len(sequence)-1):
                    X_before.append(sequence[i-1])
                    X_after.append(sequence[i+1])
                    y.append(sequence[i])
        return X_before, X_after, y

    X_before, X_after, y = create_io_pairs(sequences)

    X_before = torch.tensor(X_before, dtype=torch.long).unsqueeze(1)
    X_after = torch.tensor(X_after, dtype=torch.long).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.long)

    dataset = WordDataset(X_before, X_after, y)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    vocab_size = len(vocab) + 1
    model = WordBetweenModel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM)

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

    predicted_word = predict_between(model, "nalazi", "centru", vocab)
    print(predicted_word)
    
    


