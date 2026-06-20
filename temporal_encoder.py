from __future__ import annotations

import torch
from torch import nn


class ChannelTemporalEncoder(nn.Module):
    """
    Mean-pooling temporal encoder over peri-onset windows.

    TCN and recurrent encoders were removed for B0-Pruned-EZBackbone. Input is
    [B, S, T, C, D], output is [B, S, C, D].
    """

    def __init__(self, model_dim: int = 96, **_: object) -> None:
        super().__init__()
        self.model_dim = int(model_dim)

    def forward(
        self,
        window_embeddings: torch.Tensor,
        seizure_channel_mask: torch.Tensor | None = None,
        window_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, s, t, c, d = window_embeddings.shape
        if seizure_channel_mask is None:
            seizure_channel_mask = torch.ones((b, s, c), dtype=torch.bool, device=window_embeddings.device)
        time_mask = seizure_channel_mask[:, :, None, :].expand(b, s, t, c)
        if window_mask is not None:
            time_mask = time_mask & window_mask[:, :, :, None].expand(b, s, t, c)

        weights = time_mask.float()
        denom = weights.sum(dim=2, keepdim=True).clamp_min(1.0)
        pooled = (window_embeddings * weights.unsqueeze(-1)).sum(dim=2) / denom.squeeze(2).unsqueeze(-1)
        pooled = pooled * seizure_channel_mask.float().unsqueeze(-1)
        temporal_weights = (weights / denom).permute(0, 1, 3, 2).contiguous()
        return pooled, temporal_weights


__all__ = ["ChannelTemporalEncoder"]
