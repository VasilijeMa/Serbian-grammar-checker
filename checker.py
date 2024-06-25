import string
from utils import *

TEXTS_FILE_NAME = "texts.txt"
DICT_FILE_NAME = "dict.txt"

EMBEDDING_DIM = 64
HIDDEN_DIM = 128

if __name__ == "__main__":
    sentences = []
    dictionary = []

    letters = string.ascii_letters + " čćđšžČĆĐŠŽ"

    def preprocess_sentence(sentence):
        return ' '.join(''.join(letter for letter in sentence if letter in letters).split())

    with open(TEXTS_FILE_NAME, 'r', encoding='utf-8') as file:
        for line in file:
            lines = line.split("\t")
            sentences.append(preprocess_sentence(lines[1]))
            sentences.append(preprocess_sentence(lines[2]))

    with open(DICT_FILE_NAME, 'r', encoding='utf-8') as file:
        for line in file:
            dictionary.append(line)
    
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
    
    corrected_sentence, best_mask_position = correct_sentence_mask(model, tokenizer, "Iz danas je amognus iz", sentences)
    print(f"Corrected: {corrected_sentence} (Best mask position: {best_mask_position})")

