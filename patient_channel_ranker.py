from __future__ import annotations

import torch
from torch import nn


class PatientChannelClassifier(nn.Module):
    """
    Lightweight patient-relative channel classifier.

    Scores are NEZ probabilities for compatibility with the existing EZ
    localization reports; EZ score is 1 - p(NEZ).
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 2,
        dropout: float = 0.25,
        use_patient_relative_z: bool = True,
    ) -> None:
        super().__init__()
        dim = int(input_dim)
        self.use_patient_relative_z = bool(use_patient_relative_z)
        self.channel_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim, 1),
        )

    def forward(self, patient_channel_embedding: torch.Tensor, channel_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        h = (
            _patient_relative_zscore(patient_channel_embedding, channel_mask)
            if self.use_patient_relative_z
            else patient_channel_embedding * channel_mask.float().unsqueeze(-1)
        )
        key_padding_mask = ~channel_mask
        all_invalid = key_padding_mask.all(dim=1)
        if torch.any(all_invalid):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_invalid] = False
        context, _ = self.channel_attn(h, h, h, key_padding_mask=key_padding_mask)
        h = self.attn_norm(h + self.dropout(context))
        logits = self.classifier(h).squeeze(-1).masked_fill(~channel_mask, -1e9)
        scores = torch.sigmoid(logits)
        return {
            "logits": logits,
            "scores": scores,
            "score_nez": scores,
            "score_ez": 1.0 - scores,
        }


def _patient_relative_zscore(x: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
    mask = channel_mask.float().unsqueeze(-1)
    count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (x * mask).sum(dim=1, keepdim=True) / count
    var = (((x - mean) * mask) ** 2).sum(dim=1, keepdim=True) / count
    z = (x - mean) / torch.sqrt(var + 1e-5)
    return z * mask


__all__ = ["PatientChannelClassifier"]
