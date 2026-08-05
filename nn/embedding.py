import random

class Embedding:
    def __init__(self, vocab_size, embedding_size):
        self.embedding_size = embedding_size
        self.table = []

        for _ in range(vocab_size):
            vector = []

            for _ in range(embedding_size):
                vector.append(random.uniform(-1, 1))

            self.table.append(vector)

    def get(self, token):
        return self.table[token]