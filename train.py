import os

import torch
from tqdm import tqdm
import torchinfo

from data.tokenizer import CharTokenizer
from data.dataset import get_dataloader
from config import dataset_config_base, model_config_base, trainer_config_base, TrainerConfig  # noqa
from model import PoetryGPT


def evaluate(model, test_dataloader, config: TrainerConfig):
    device = model.config.device
    model.eval()
    with torch.no_grad():
        loss = []
        for context, target in test_dataloader:
            context, target = context.to(device), target.to(device)  # noqa
            logits = model(context)
            loss.append(config.loss_fn(logits.transpose(1, 2), target).item())
    return sum(loss) / len(loss)


def train(model, train_dataloader, test_dataloader, config: TrainerConfig):
    device = model.config.device
    optimizer = config.optimizer(model.parameters(), lr=config.learning_rate)
    model.train()
    for epoch in range(config.n_epochs):
        step = 0
        for context, target in tqdm(
            train_dataloader,
            desc=f"epoch: {epoch}/{config.n_epochs}"
        ):
            context, target = context.to(device), target.to(device)
            logits = model(context)
            loss = config.loss_fn(logits.transpose(1, 2), target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1

        if epoch % config.eval_interval == 0:
            model.eval()
            test_loss = evaluate(model, test_dataloader, config)
            print(f"epoch: {epoch}: train loss {loss.item()}, test loss {test_loss}")  # noqa
            model.train()

        if epoch % config.checkpoint_interval == 0:
            if not os.path.exists(config.checkpoint_path.split("/")[0]):
                os.makedirs(config.checkpoint_path.split("/")[0])
            torch.save(model.state_dict(), config.checkpoint_path.format(epoch=epoch))

    if not os.path.exists(config.model_path.split("/")[0]):
        os.makedirs(config.model_path.split("/")[0])
    torch.save(model.state_dict(), config.model_path)


if __name__ == "__main__":
    train_dataloader = get_dataloader("train", dataset_config_base)
    test_dataloader = get_dataloader("test", dataset_config_base)

    tokenizer = CharTokenizer()
    tokenizer.fit(dataset_config_base.data_path, dataset_config_base.min_occurance)

    model = PoetryGPT(tokenizer, model_config_base)
    torchinfo.summary(model)
    print(f"training on {next(model.parameters()).device}")
    train(model, train_dataloader, test_dataloader, trainer_config_base)
