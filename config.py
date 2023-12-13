import torch
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    batch_size: int
    seq_len: int
    train_split: float
    data_path: str


@dataclass
class TrainerConfig:
    n_epochs: int
    learning_rate: float
    optimizer: torch.optim
    loss_fn: torch.nn
    eval_interval: int
    checkpoint_interval: int
    checkpoint_path: str
    model_path: str


@dataclass
class ModelConfig:
    batch_size: int
    emb_dim: int
    seq_len: int
    n_head: int
    head_dim: int
    n_layer: int
    dropout: float
    device: torch.cuda.device = 'cuda' if torch.cuda.is_available() else 'cpu'


dataset_config_base = DatasetConfig(
    batch_size=16,
    seq_len=32,
    train_split=0.9,
    data_path="data/poems.txt",
)

trainer_config_base = TrainerConfig(
    n_epochs=5000,
    learning_rate=1e-3,
    optimizer=torch.optim.AdamW,
    loss_fn=torch.nn.CrossEntropyLoss(),
    eval_interval=2000,
    checkpoint_interval=1,
    checkpoint_path="checkpoints/",
    model_path="models/"
)

model_config_base = ModelConfig(
    batch_size=dataset_config_base.batch_size,
    emb_dim=64,
    seq_len=dataset_config_base.seq_len,
    n_head=4,
    head_dim=64 // 4,
    n_layer=4,
    dropout=0.2,
)
