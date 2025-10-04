from .Transformer import TransformerBlock
import torch
import torch.nn as nn

class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, n_blocks: int, n_heads: int, dropout: float):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(context_length, d_model)
        transformers = [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_blocks)]
        self.transformer_blocks = nn.Sequential(*transformers)
        self.final_layernorm = nn.LayerNorm(d_model)
        self.linear_layer = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        B, T = x.shape

        wrd_embeddings = self.word_embeddings(x)
        positions = torch.arange(T, device=x.device)
        pos_embeddings = self.pos_embeddings(positions).unsqueeze(0)
        embeddings = wrd_embeddings + pos_embeddings

        out = self.transformer_blocks(embeddings)
        out_norm = self.final_layernorm(out)
        logits = self.linear_layer(out_norm)
        logits = self.softmax(logits)

        return logits