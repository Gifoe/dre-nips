from __future__ import annotations

from typing import Any

import torch
from torch import nn

from graph_spectral_encoder import WindowGraphSpectralEncoder
from patient_channel_ranker import PatientChannelClassifier
from seizure_aggregator import CrossSeizureMILAggregator
from temporal_encoder import ChannelTemporalEncoder


class NeuroEZCModel(nn.Module):
    """B0-Pruned patient-level spectral/classical model for EZ/NEZ localization."""

    def __init__(self, args: Any | None = None) -> None:
        super().__init__()
        self.model_dim = int(getattr(args, "model_dim", 32) if args is not None else 32)
        dropout = float(getattr(args, "dropout", 0.40) if args is not None else 0.40)
        num_heads = int(getattr(args, "num_heads", 2) if args is not None else 2)
        use_channel_attention = bool(getattr(args, "use_channel_attention", True) if args is not None else True)

        self.b0_encoder = WindowGraphSpectralEncoder(
            model_dim=self.model_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_channel_attention=use_channel_attention,
        )
        self.fusion_norm = nn.LayerNorm(self.model_dim)
        self.temporal_encoder = ChannelTemporalEncoder(model_dim=self.model_dim)
        self.seizure_aggregator = CrossSeizureMILAggregator(model_dim=self.model_dim)
        self.channel_classifier = PatientChannelClassifier(
            input_dim=self.seizure_aggregator.output_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_patient_relative_z=bool(getattr(args, "use_patient_relative_z", True) if args is not None else True),
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        seizure_channel_mask = batch["seizure_channel_mask"]
        h_b0 = self.b0_encoder(batch["b0_features"], None, seizure_channel_mask)
        fused = self.fusion_norm(h_b0)
        seizure_channel_embedding, temporal_weights = self.temporal_encoder(
            fused,
            seizure_channel_mask,
            window_mask=batch.get("window_mask"),
        )
        patient_channel_embedding, seizure_weights = self.seizure_aggregator(
            seizure_channel_embedding,
            batch["seizure_mask"],
            seizure_channel_mask,
        )
        output = self.channel_classifier(patient_channel_embedding, batch["channel_mask"])
        output.update(
            {
                "patient_channel_embedding": patient_channel_embedding,
                "task_embedding": patient_channel_embedding,
                "seizure_channel_embedding": seizure_channel_embedding,
                "temporal_attention": temporal_weights,
                "seizure_attention": seizure_weights,
            }
        )
        return output


__all__ = ["NeuroEZCModel"]
