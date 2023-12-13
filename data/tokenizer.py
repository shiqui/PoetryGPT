class CharTokenizer:
    def __init__(self):
        self.token_to_index = {}
        self.index_to_token = {}
        self.vocab_size = 0

    def fit(self, text):
        tokens = ["<", "_", ">"] + sorted(list(set(text)))
        # <: start of sequence
        # _: padding
        # >: end of sequence
        self.vocab_size = len(tokens)
        self.token_to_index = {c: i for i, c in enumerate(tokens)}
        self.index_to_token = {i: c for i, c in enumerate(tokens)}

    def encode(self, s: str):
        return [self.token_to_index[c] for c in s]

    def decode(self, seq: [int]):
        return "".join([self.index_to_token[i] for i in seq])


if __name__ == "__main__":
    tokenizer = CharTokenizer()
    with open("data/poems.txt") as f:
        text = f.read()
    tokenizer.fit(text)
    from itertools import islice
    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"vocab: {dict(islice(tokenizer.index_to_token.items(), 10))} ...")
