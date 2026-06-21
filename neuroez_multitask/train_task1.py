from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def task1_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    labels_nez = batch["labels_nez"].float()
    mask = batch["channel_mask"].bool() & (labels_nez >= 0.0)
    if not torch.any(mask):
        return outputs["nez_logits"].sum() * 0.0
    return F.binary_cross_entropy_with_logits(outputs["nez_logits"][mask], labels_nez[mask])


def task1_prediction_rows(
    *,
    subject_id: str,
    center: str,
    fold: int,
    channel_names: Sequence[str],
    labels_nez: Sequence[float],
    nez_prob: Sequence[float],
    channel_mask: Sequence[bool],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, label, prob, valid in zip(channel_names, labels_nez, nez_prob, channel_mask):
        if not valid or float(label) < 0.0:
            continue
        rows.append(
            {
                "subject_id": subject_id,
                "channel_name": str(name),
                "label_nez": float(label),
                "label_ez": float(1.0 - float(label)),
                "nez_prob": float(prob),
                "ez_prob": float(1.0 - float(prob)),
                "fold": int(fold),
                "center": center,
            }
        )
    return rows
