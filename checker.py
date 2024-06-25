import string

TEXTS_FILE_NAME = "texts.txt"
DICT_FILE_NAME = "dict.txt"

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

    print(len(dictionary))
    print(len(sentences))