from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .causal_graph_encoder import DelayAwareDirectedGraphEncoder
from .outcome_head import EZHead, LearnedOutcomeAttentionReadout, masked_mean, masked_std
from .physics_encoder import B0Encoder, GatedPhysicsFusion, PhysicsEncoder


class PGCSEEGModel(nn.Module):
    def __init__(
        self,
        model_dim: int = 64,
        *,
        topology_dim: int = 8,
        dropout: float = 0.1,
        use_physics_branch: bool = True,
        use_causal_graph: bool = True,
        use_causal_node_features: bool = True,
        use_delay: bool = True,
        directed_graph: bool = True,
        random_graph: bool = False,
        fusion_type: str = "gated",
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.use_physics_branch = use_physics_branch
        self.use_causal_graph = use_causal_graph
        self.use_causal_node_features = use_causal_node_features
        self.b0_encoder = B0Encoder(model_dim=model_dim, dropout=dropout)
        self.physics_encoder = PhysicsEncoder(model_dim=model_dim, dropout=dropout)
        self.physics_fusion = GatedPhysicsFusion(model_dim=model_dim, fusion_type=fusion_type)
        self.causal_graph_encoder = DelayAwareDirectedGraphEncoder(
            model_dim=model_dim,
            use_delay=use_delay,
            directed_graph=directed_graph,
            random_graph=random_graph,
        )
        self.causal_node_projection = nn.LazyLinear(model_dim)
        self.ez_head = EZHead(input_dim=model_dim * 2, dropout=dropout)
        self.outcome_head = LearnedOutcomeAttentionReadout(embedding_dim=model_dim * 2, topology_dim=topology_dim, dropout=dropout)

    def _temporal_pool(self, h: torch.Tensor, window_mask: torch.Tensor) -> torch.Tensor:
        mask = window_mask[:, :, :, None].expand(h.shape[0], h.shape[1], h.shape[2], h.shape[3])
        return masked_mean(h, mask, dim=2)

    def _seizure_aggregate(self, seizure_channel_embedding: torch.Tensor, seizure_channel_mask: torch.Tensor) -> torch.Tensor:
        mean = masked_mean(seizure_channel_embedding, seizure_channel_mask, dim=1)
        std = masked_std(seizure_channel_embedding, seizure_channel_mask, dim=1)
        return torch.cat([mean, std], dim=-1)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        b0_features = batch["b0_features"].float()
        h = self.b0_encoder(b0_features)
        if self.use_physics_branch:
            h_phys = self.physics_encoder(batch["physics_features"].float())
            h = self.physics_fusion(h, h_phys)
        if self.use_causal_graph:
            causal_node = batch["causal_node_features"].float()
            if self.use_causal_node_features:
                causal_node_projected = self.causal_node_projection(causal_node)
            else:
                causal_node_projected = torch.zeros_like(h)
            h = self.causal_graph_encoder(
                h,
                batch["causal_adjacency"].float(),
                batch["causal_delay"].float(),
                causal_node_projected,
            )
        seizure_channel_embedding = self._temporal_pool(h, batch["window_mask"].bool())
        patient_channel_embedding = self._seizure_aggregate(
            seizure_channel_embedding,
            batch["seizure_channel_mask"].bool(),
        )
        channel_mask = batch["channel_mask"].bool()
        nez_logits = self.ez_head(patient_channel_embedding)
        nez_prob = torch.sigmoid(nez_logits)
        causal_summary = batch["causal_node_features"].float().mean(dim=(1, 2))
        physics_summary = batch["physics_features"].float().mean(dim=(1, 2))
        outcome_logit, attention = self.outcome_head(
            patient_channel_embedding,
            channel_mask,
            causal_channel_summary=causal_summary,
            physics_channel_summary=physics_summary,
            topology_features=batch["topology_features"].float(),
        )
        outcome_prob = torch.sigmoid(outcome_logit)
        return {
            "nez_logits": nez_logits,
            "nez_prob": nez_prob,
            "ez_logits": -nez_logits,
            "ez_prob": 1.0 - nez_prob,
            "outcome_logit": outcome_logit,
            "outcome_prob": outcome_prob,
            "patient_channel_embedding": patient_channel_embedding,
            "outcome_attention": attention,
        }
