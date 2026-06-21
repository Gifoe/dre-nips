from __future__ import annotations

import torch
from torch import nn


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    total = torch.sum(values * weights, dim=dim)
    denom = torch.sum(weights, dim=dim).clamp_min(1.0)
    return total / denom


def masked_std(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mean = masked_mean(values, mask, dim=dim)
    expanded = mean.unsqueeze(dim)
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    var = torch.sum(((values - expanded) ** 2) * weights, dim=dim) / torch.sum(weights, dim=dim).clamp_min(1.0)
    return torch.sqrt(var.clamp_min(0.0) + 1e-6)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probs = torch.softmax(masked, dim=dim)
    return torch.where(mask, probs, torch.zeros_like(probs))


class EZHead(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
        )

    def forward(self, patient_channel_embedding: torch.Tensor) -> torch.Tensor:
        return self.net(patient_channel_embedding).squeeze(-1)


class LearnedOutcomeAttentionReadout(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        topology_dim: int = 0,
        dropout: float = 0.1,
        *,
        readout_type: str = "attention",
        use_topology_features: bool = True,
    ) -> None:
        super().__init__()
        readout = readout_type.lower().strip()
        if readout not in {"global", "attention"}:
            raise ValueError(f"Unsupported outcome readout_type={readout_type!r}.")
        self.readout_type = readout
        self.use_topology_features = bool(use_topology_features)
        self.attention = nn.Sequential(
            nn.LazyLinear(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )
        self.outcome_head = nn.Sequential(
            nn.LazyLinear(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(
        self,
        patient_channel_embedding: torch.Tensor,
        channel_mask: torch.Tensor,
        *,
        causal_channel_summary: torch.Tensor,
        physics_channel_summary: torch.Tensor,
        topology_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_mean = masked_mean(patient_channel_embedding, channel_mask, dim=1)
        z_std = masked_std(patient_channel_embedding, channel_mask, dim=1)
        if self.readout_type == "attention":
            query = torch.cat([patient_channel_embedding, causal_channel_summary, physics_channel_summary], dim=-1)
            attention_logits = self.attention(query).squeeze(-1)
            alpha = masked_softmax(attention_logits, channel_mask, dim=-1)
            z_attn = torch.sum(alpha.unsqueeze(-1) * patient_channel_embedding, dim=1)
            parts = [z_attn, z_mean, z_std]
        else:
            alpha = torch.zeros_like(channel_mask, dtype=patient_channel_embedding.dtype)
            parts = [z_mean, z_std]
        if self.use_topology_features:
            parts.append(topology_features)
        z_patient = torch.cat(parts, dim=-1)
        return self.outcome_head(z_patient).squeeze(-1), alpha
