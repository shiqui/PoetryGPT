import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class Head(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.emb_dim = config.emb_dim
        self.head_dim = config.head_dim
        self.device = config.device

        # projection to a lower dimension
        self.Wq = nn.Linear(self.emb_dim, self.head_dim, bias=False)
        self.Wk = nn.Linear(self.emb_dim, self.head_dim, bias=False)
        self.Wv = nn.Linear(self.emb_dim, self.head_dim, bias=False)

    def forward(self, x):
        Q = self.Wq(x)  # (batch_size, seq_len, emb_dim)
        K = self.Wk(x)  # (batch_size, seq_len, emb_dim)
        V = self.Wv(x)  # (batch_size, seq_len, emb_dim)
        # this head is for self attention
        # Q, K, V are garanteed to have the same sequence length

        similarities = torch.matmul(Q, K.transpose(-2, -1))
        similarities = similarities / self.emb_dim ** 0.5
        # (batch_size, seq_len, seq_len)
        # row i: how related key i is to each key

        # mask so that target i only pays attention to context[:i]
        seq_len = similarities.shape[1]
        tril = torch.tril(torch.ones(seq_len, seq_len))
        tril = tril.to(self.device)
        similarities = similarities.masked_fill(tril == 0, float('-inf'))

        attention = F.softmax(similarities, dim=-1)

        # use attention as weights, weighted sum of values
        return torch.matmul(attention, V)


class MultiHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(config) for _ in range(config.n_head)]
        )
        self.proj = nn.Linear(config.emb_dim, config.emb_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # each head is in a lower dimension
        # concat all heads and project back to original dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FFN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.emb_dim, 4 * config.emb_dim),
            nn.ReLU(),
            nn.Linear(4 * config.emb_dim, config.emb_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        # expand and shrink back, with relu and dropout
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attention = MultiHead(config)
        self.ffn = FFN(config)
        self.ln1 = nn.LayerNorm(config.emb_dim)
        self.ln2 = nn.LayerNorm(config.emb_dim)

    def forward(self, x):
        # residual connection and layer norm after each sublayer
        x = x + self.self_attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class PoetryGPT(nn.Module):
    def __init__(self, tokenizer, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        self.embedding_table = nn.Embedding(
            tokenizer.vocab_size, config.emb_dim
        )
        self.position_encoding = nn.Embedding(
            config.seq_len, config.emb_dim
        )
        self.blocks = nn.Sequential(
            *[DecoderBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.emb_dim)
        self.lm_head = nn.Linear(config.emb_dim, tokenizer.vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape

        tok_emb = self.embedding_table(x)  # (batch_size, seq_len, emb_dim)

        pos_emb = self.position_encoding(
            torch.arange(seq_len, device=self.config.device)
        )  # (seq_len, emb_dim)
        x = tok_emb + pos_emb
        x = self.blocks(x)  # (batch_size, seq_len, emb_dim)
        x = self.ln_f(x)  # (batch_size, seq_len, emb_dim)
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)

        return logits

    def generate(self, context):
        self.eval()
        context = context[:, -self.config.seq_len:]
        logits = self(context[:, -self.config.seq_len:])  # crop if too long
        # (batch_size, seq_len, vocab_size)
        logits = logits[:, -1, :]  # take last token only
        # (batch_size, vocab_size)
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token
