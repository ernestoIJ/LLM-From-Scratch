import torch
import torch.nn as nn
import math
from typing import Tuple

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int, dropout: float):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)
        self.scale = 1 / math.sqrt(d_k)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)

    
    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores =(Q @ K.transpose(-2, -1)) * self.scale

        T = K.shape[1]
        upper_mask = torch.triu(torch.ones((T, T), dtype=torch.bool, device=scores.device), diagonal=1)
        masked_scores = scores.masked_fill(upper_mask, -torch.inf)
        
        attn_probs = self.softmax(masked_scores)
        attn_probs = self.dropout(attn_probs)

        context = attn_probs @ V

        return (context, attn_probs) if return_attn else context
