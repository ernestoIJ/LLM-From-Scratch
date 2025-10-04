from .MultiHeadAttention import MultiHeadAttention
from .MLP import VanillaMLP
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attention_heads = MultiHeadAttention(d_model, n_heads, dropout)
        self.mlp = VanillaMLP(d_model, dropout)
        self.first_normalization = nn.LayerNorm(d_model)
        self.second_normalization = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        x = x + self.attention_heads(self.first_normalization(x))
        x = x + self.mlp(self.second_normalization(x))

        return x
