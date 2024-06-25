import string
from utils import *

TEXTS_FILE_NAME = "texts.txt"
VOCAB_FILE_NAME = "dict.txt"
LETTERS = string.ascii_letters + " čćđšžČĆĐŠŽ"

EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_EPOCHS = 5


if __name__ == "__main__":
    sentences = []
    vocabulary = []

    with open(TEXTS_FILE_NAME, 'r', encoding='utf-8') as file:
        for line in file:
            lines = line.split("\t")
            sentences.append(preprocess_sentence(lines[1].strip(), LETTERS))
            sentences.append(preprocess_sentence(lines[2].strip(), LETTERS))

    with open(VOCAB_FILE_NAME, 'r', encoding='utf-8') as file:
        for line in file:
            vocabulary.append(line.strip())
    
    tokenizer = Tokenizer()
    tokenizer.fit(sentences)

    encoded_sentences = [torch.tensor(tokenizer.encode(sentence)) for sentence in sentences]
    padded_sentences = pad_sequence(encoded_sentences, batch_first=True, padding_value=tokenizer.word2idx['<PAD>'])

    dataset = SentenceDataset(padded_sentences)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    vocab_size = tokenizer.vocab_size

    model = LanguageModel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # for epoch in range(NUM_EPOCHS):
    #     for inputs, targets in dataloader:
    #         outputs = model(inputs)
    #         loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {loss.item()}')

    # corrected_sentence, best_mask_position = correct_sentence_mask(model, tokenizer, "Iz danas je amognus iz", sentences)
    # print(f"Corrected: {corrected_sentence} (Best mask position: {best_mask_position})")

    prompt = preprocess_sentence("Ovo je rekenica", LETTERS)
    for combination in spellcheck_sentence(prompt, vocabulary):
        print(' '.join(word for word in combination))

