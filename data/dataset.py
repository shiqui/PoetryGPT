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
    tokenizer = CharTokenizer()
    with open(config.data_path) as f:
        text = f.read()
    tokenizer.fit(text)
    poems = text.split("\n\n")
    data = []
    for poem in poems:
        if len(poem) < config.seq_len - 1:
            padded_poem = f"<{poem}{'_' * (config.seq_len - len(poem) - 1)}>"
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
