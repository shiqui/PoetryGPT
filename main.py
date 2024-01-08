import torch

from model import PoetryGPT
from data.tokenizer import CharTokenizer
from config import dataset_config_base, model_config_base, trainer_config_base, TrainerConfig  # noqa


def generate(model, max_len=100, context=None, temperature=1.0):
    temperature = temperature if temperature > 0 else 1.0

    context = "<" if context is None else f"<{context}"
    encoded = torch.tensor(
        tokenizer.encode(context),
        requires_grad=False,
        device=model.config.device
    ).unsqueeze(0)

    for i in range(max_len):
        token = model.generate(encoded, temperature=temperature)
        encoded = torch.cat([encoded, token], dim=-1)
        if token == tokenizer.token_to_index[">"]:
            break
    
    return tokenizer.decode(
        [t for t in encoded[0].tolist() if t not in (0, 1, 2, 3)]
    )


if __name__ == "__main__":
    tokenizer = CharTokenizer()
    tokenizer.fit(dataset_config_base.data_path, dataset_config_base.min_occurance)

    model = PoetryGPT(tokenizer, model_config_base)
    model.load_state_dict(torch.load("models/base.pth", map_location=model_config_base.device))
    print(generate(model, context="", temperature=1.5))