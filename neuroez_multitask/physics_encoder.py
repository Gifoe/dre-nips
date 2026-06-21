from __future__ import annotations

import torch
from torch import nn


class B0Encoder(nn.Module):
    def __init__(self, model_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class PhysicsEncoder(nn.Module):
    def __init__(self, model_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class GatedPhysicsFusion(nn.Module):
    def __init__(self, model_dim: int = 64, fusion_type: str = "gated") -> None:
        super().__init__()
        self.fusion_type = fusion_type
        self.gate = nn.Sequential(nn.Linear(model_dim * 2, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        self.concat = nn.Sequential(nn.Linear(model_dim * 2, model_dim), nn.GELU(), nn.LayerNorm(model_dim))

    def forward(self, h_b0: torch.Tensor, h_phys: torch.Tensor) -> torch.Tensor:
        merged = torch.cat([h_b0, h_phys], dim=-1)
        if self.fusion_type == "concat":
            return self.concat(merged)
        gate = torch.sigmoid(self.gate(merged))
        return h_b0 + gate * h_phys
