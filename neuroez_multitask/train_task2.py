from __future__ import annotations

import torch
import torch.nn.functional as F


def task2_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    labels = batch["outcome_label"].float()
    mask = batch["outcome_mask"].bool()
    if not torch.any(mask):
        return outputs["outcome_logit"].sum() * 0.0
    return F.binary_cross_entropy_with_logits(outputs["outcome_logit"][mask], labels[mask], pos_weight=pos_weight)
