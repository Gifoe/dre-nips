from __future__ import annotations

import torch
from torch import nn


class WindowGraphSpectralEncoder(nn.Module):
    """
    Pruned window-level spectral/classical encoder.

    The class name is retained for import compatibility, but graph adjacency
    message passing has been removed. Input features are [B, S, T, C, F] and the
    output is [B, S, T, C, D].
    """

    def __init__(
        self,
        model_dim: int = 96,
        num_heads: int = 4,
        dropout: float = 0.25,
        use_channel_attention: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.use_channel_attention = bool(use_channel_attention)
        self.feature_mlp = nn.Sequential(
            nn.LazyLinear(self.model_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.model_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
        )
        self.channel_attn = nn.MultiheadAttention(
            embed_dim=self.model_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.model_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        seizure_channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del adjacency
        h = self.feature_mlp(features)
        if not self.use_channel_attention:
            return h

        b, s, t, c, d = h.shape
        flat_h = h.reshape(b * s * t, c, d)
        key_padding_mask = None
        if seizure_channel_mask is not None:
            invalid = ~seizure_channel_mask[:, :, None, :].expand(b, s, t, c)
            key_padding_mask = invalid.reshape(b * s * t, c)
            all_invalid = key_padding_mask.all(dim=1)
            if torch.any(all_invalid):
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_invalid] = False
        attn_h, _ = self.channel_attn(flat_h, flat_h, flat_h, key_padding_mask=key_padding_mask)
        return self.attn_norm(flat_h + self.dropout(attn_h)).reshape(b, s, t, c, d)


__all__ = ["WindowGraphSpectralEncoder"]
