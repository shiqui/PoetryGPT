import torch

from model import PoetryGPT
from data.tokenizer import CharTokenizer
from config import dataset_config_base, model_config_base, trainer_config_base, TrainerConfig  # noqa


def generate(model, max_len=100, context=None):
    context = "<" if context is None else f"<{context}"
    encoded = torch.tensor(
        tokenizer.encode(context),
        requires_grad=False,
        device=model.config.device
    ).unsqueeze(0)

    for i in range(max_len):
        token = model.generate(encoded)
        encoded = torch.cat([encoded, token], dim=-1)
        if token == tokenizer.token_to_index[">"]:
            break

    return tokenizer.decode(encoded[0][1:-1].tolist())


if __name__ == "__main__":
    tokenizer = CharTokenizer()
    with open(dataset_config_base.data_path) as f:
        text = f.read()
    tokenizer.fit(text)

    model = PoetryGPT(tokenizer, model_config_base)
    model.load_state_dict(torch.load("checkpoints/model_18.pth"))

    print(generate(model, context=""))
