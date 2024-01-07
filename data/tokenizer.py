import os
import json


class CharTokenizer:
    def __init__(self):
        self.vocab = {"<", "_", ">", "O"}
        # <: start of sequence
        # _: padding
        # >: end of sequence
        # O: out of vocabulary
        self.token_to_index = {c: i for i, c in enumerate(self.vocab)}
        self.index_to_token = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def fit(self, directory="data/raw", min_occurance=5):
        paragraphs = []
        for filename in os.listdir(directory):
            with open(os.path.join(directory, filename), "r") as f:
                data = json.load(f)
            paragraphs += ["\n".join(item["paragraphs"]) for item in data]
        self.fit_from_string("".join(paragraphs), min_occurance)

    def fit_from_string(self, s, min_occurance=5):
        occurance = {}
        for c in s:
            if c not in occurance:
                occurance[c] = 1
            else:
                occurance[c] += 1
        for c in occurance:
            if occurance[c] >= min_occurance and c not in self.vocab:
                self.vocab.add(c)
                self.token_to_index[c] = self.vocab_size
                self.index_to_token[self.vocab_size] = c
                self.vocab_size += 1

    def encode(self, s: str):
        OOV_token = self.token_to_index["O"]
        return [self.token_to_index.get(c, OOV_token) for c in s]

    def decode(self, seq: [int]):
        return "".join([self.index_to_token[i] for i in seq])


if __name__ == "__main__":
    import json
    tokenizer = CharTokenizer()
    tokenizer.fit(min_occurance=10)

    print(len(tokenizer.vocab))