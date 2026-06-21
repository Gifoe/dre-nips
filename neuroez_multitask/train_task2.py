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


def estimate_task2_pos_weight(cache_payload: dict, train_subjects: list[str] | tuple[str, ...]) -> torch.Tensor:
    outcome_index = cache_payload.get("outcome_index", {})
    positives = 0
    negatives = 0
    for sid in train_subjects:
        value = outcome_index.get(str(sid), {}).get("success_failure")
        if value is None:
            continue
        if int(value) == 1:
            positives += 1
        else:
            negatives += 1
    if positives <= 0 or negatives <= 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(float(negatives) / float(positives), dtype=torch.float32)
