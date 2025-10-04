import torch
import torch.nn as nn

class VanillaMLP(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(p=dropout)
        )
    
    def forward(self, x: torch.Tensor):
        return self.ffn(x)