from .SingleHeadSelfAttention import ScaledDotProductAttention
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "model dimension (d_model) must be divisible by the number of heads (n_heads)"
        d_k = d_model // n_heads
        self.heads = nn.ModuleList(
            [ScaledDotProductAttention(d_model, d_k, dropout) for _ in range(n_heads)]
        )
    
    def forward(self, x: torch.Tensor):
        context = torch.cat([head(x) for head in self.heads], dim=-1)
        return context
    