from __future__ import annotations

import torch
from torch import nn


def row_normalize(matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    denom = matrix.sum(dim=-1, keepdim=True).clamp_min(eps)
    return matrix / denom


class DelayAwareDirectedGraphEncoder(nn.Module):
    def __init__(
        self,
        model_dim: int = 64,
        *,
        use_delay: bool = True,
        directed_graph: bool = True,
        random_graph: bool = False,
    ) -> None:
        super().__init__()
        self.use_delay = use_delay
        self.directed_graph = directed_graph
        self.random_graph = random_graph
        self.w_out = nn.Linear(model_dim, model_dim)
        self.w_in = nn.Linear(model_dim, model_dim)
        self.delay_encoder = nn.Sequential(nn.LazyLinear(model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        self.update = nn.Sequential(
            nn.LazyLinear(model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.gate = nn.Sequential(nn.Linear(model_dim * 2, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))

    def forward(
        self,
        h: torch.Tensor,
        causal_adjacency: torch.Tensor,
        causal_delay: torch.Tensor,
        causal_node_features: torch.Tensor,
    ) -> torch.Tensor:
        adjacency = causal_adjacency
        if not self.directed_graph:
            adjacency = 0.5 * (adjacency + adjacency.transpose(-1, -2))
        if self.random_graph:
            adjacency = torch.rand_like(adjacency) * (adjacency > 0).to(adjacency.dtype)
        a_out = row_normalize(adjacency)
        a_in = row_normalize(adjacency.transpose(-1, -2))
        m_out = torch.matmul(a_out, self.w_out(h))
        m_in = torch.matmul(a_in, self.w_in(h))
        if self.use_delay:
            delay_summary = self.delay_encoder(causal_delay.mean(dim=-1, keepdim=True))
        else:
            delay_summary = torch.zeros_like(h)
        h_causal = self.update(torch.cat([h, m_out, m_in, causal_node_features, delay_summary], dim=-1))
        gate = torch.sigmoid(self.gate(torch.cat([h, h_causal], dim=-1)))
        return h + gate * h_causal
