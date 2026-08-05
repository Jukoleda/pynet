class Tokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}

    def fit(self, text):
        words = sorted(set(text.split()))

        for i, word in enumerate(words):
            self.word_to_id[word] = i
            self.id_to_word[i] = word

    def encode(self, text):
        return [self.word_to_id[word] for word in text.split()]

    def decode(self, tokens):
        return " ".join(self.id_to_word[token] for token in tokens)