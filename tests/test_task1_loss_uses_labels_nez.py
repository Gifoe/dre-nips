from __future__ import annotations

import torch

from neuroez_multitask.train_task1 import task1_loss


def test_task1_loss_uses_nez_positive_labels_not_ez_labels():
    outputs = {"nez_logits": torch.tensor([[0.0, 2.0, -2.0]])}
    batch = {
        "labels_nez": torch.tensor([[0.0, 1.0, -1.0]]),
        "labels_ez": torch.tensor([[1.0, 0.0, -1.0]]),
        "channel_mask": torch.tensor([[True, True, False]]),
    }

    got = task1_loss(outputs, batch)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([0.0, 2.0]),
        torch.tensor([0.0, 1.0]),
    )
    assert torch.allclose(got, expected)
