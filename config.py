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
    n_epochs: int = 5000
    learning_rate: float = 1e-3
    eval_interval: int = 100
    checkpoint_interval: int = 500
    checkpoint_path: str = "checkpoints/"
    model_path: str = "models/"
    device: torch.cuda.device = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class ModelConfig:
    batch_size: int
    emb_dim: int
    seq_len: int
    n_head: int
    head_dim: int
    n_layer: int
    dropout: float


dataset_config_base = DatasetConfig(
    batch_size=16,
    seq_len=32,
    train_split=0.9,
    data_path="data/poems.txt",
)

trainer_config_base = TrainerConfig()

model_config_base = ModelConfig(
    batch_size=dataset_config_base.batch_size,
    emb_dim=64,
    seq_len=dataset_config_base.seq_len,
    n_head=4,
    head_dim=64 // 4,
    n_layer=4,
    dropout=0.2,
)
