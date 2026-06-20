from __future__ import annotations

import torch
from torch import nn


class CrossSeizureMILAggregator(nn.Module):
    """
    Cross-seizure mean plus variability aggregation for each channel.

    Output dimension is 2 * model_dim: concatenated masked mean and masked std.
    """

    def __init__(self, model_dim: int = 96, **_: object) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.output_dim = self.model_dim * 2

    def forward(
        self,
        seizure_channel_embedding: torch.Tensor,
        seizure_mask: torch.Tensor,
        seizure_channel_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = seizure_mask[:, :, None] & seizure_channel_mask
        weights = valid.float()
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (seizure_channel_embedding * weights.unsqueeze(-1)).sum(dim=1) / denom.squeeze(1).unsqueeze(-1)
        centered = (seizure_channel_embedding - mean[:, None, :, :]) * weights.unsqueeze(-1)
        var = centered.square().sum(dim=1) / denom.squeeze(1).unsqueeze(-1)
        std = torch.sqrt(var.clamp_min(1e-8))
        embedding = torch.cat([mean, std], dim=-1)
        embedding = embedding * valid.any(dim=1).float().unsqueeze(-1)
        seizure_weights = (weights / denom).permute(0, 2, 1).contiguous()
        return embedding, seizure_weights


__all__ = ["CrossSeizureMILAggregator"]
