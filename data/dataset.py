import os
import json


import torch
from torch.utils.data import Dataset, DataLoader

from data.tokenizer import CharTokenizer
from config import DatasetConfig


class PoemDataset(Dataset):
    def __init__(self, data, config: DatasetConfig):
        self.data = data
        self.seq_len = config.seq_len
        self.size = len(data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item[:-1]
        target = item[1:]
        return context, target


def get_dataloader(split, config: DatasetConfig):
    # Fit tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit(config.data_path, config.min_occurance)

    # Load data
    poems = []
    for filename in os.listdir(config.data_path):
        with open(os.path.join(config.data_path, filename), "r") as f:
            data = json.load(f)
        poems += ["\n".join(item["paragraphs"]) for item in data]

    # Encode data
    data = []

    for poem in poems:
        if any(c not in tokenizer.vocab for c in poem):
            # tokenizer vocab exclues rare characters
            # this should help with overfitting
            continue

        if len(poem) <= config.seq_len - 2:
            padded_poem = f"<{poem}>{'_' * (config.seq_len - len(poem) - 1)}"
            data.append(
                torch.tensor(tokenizer.encode(padded_poem), dtype=torch.long)
            )
        else:
            padded_poem = f"<{poem}>"
            for i in range(len(padded_poem) - config.seq_len):
                data.append(
                    torch.tensor(
                        tokenizer.encode(padded_poem[i:i+config.seq_len+1]),
                        dtype=torch.long
                    )
                )

    split_idx = int(config.train_split * len(data))
    data = data[:split_idx] if split == "train" else data[split_idx:]
    dataset = PoemDataset(data, config)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        drop_last=True
    )
