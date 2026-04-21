from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_arg(args: Any, name: str, default: Any) -> Any:
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


class TemporalChannelEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_model: int,
        *,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        padding = max(1, kernel_size // 2)
        self.input_proj = nn.Linear(in_dim, d_model)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )
        self.window_head = nn.Linear(d_model, 1)

    def forward(
        self,
        x_feat: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # x_feat: [W, C, F]
        if x_feat.dim() != 3:
            raise ValueError("TemporalChannelEncoder expects x_feat with shape [W, C, F].")

        projected = self.input_proj(x_feat)  # [W, C, D]
        seq = projected.permute(1, 2, 0).contiguous()  # [C, D, W]
        seq = self.temporal_conv(seq)
        seq = seq.permute(0, 2, 1).contiguous()  # [C, W, D]
        hidden, _ = self.temporal_gru(seq)

        attn_logits = self.attn(hidden).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        channel_emb = torch.sum(hidden * attn_weights, dim=1)
        window_logits = self.window_head(hidden).squeeze(-1).transpose(0, 1).contiguous()  # [W, C]

        if channel_mask is not None:
            channel_emb = channel_emb * channel_mask.unsqueeze(-1).float()
            window_logits = window_logits * channel_mask.unsqueeze(0).float()

        return {
            "channel_emb": channel_emb,
            "window_hidden": hidden,
            "window_logits": window_logits,
        }


class LocalGraphEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        d_model: int,
        *,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.update = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def _message_passing(
        self,
        node_repr: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return node_repr

        src = edge_index[0].long()
        dst = edge_index[1].long()
        edge_repr = self.edge_proj(edge_attr)

        messages = node_repr[src] + edge_repr
        aggregated = node_repr.new_zeros(node_repr.shape)
        aggregated.index_add_(0, dst, messages)

        degree = node_repr.new_zeros((node_repr.size(0), 1))
        degree.index_add_(0, dst, torch.ones((dst.shape[0], 1), device=node_repr.device, dtype=node_repr.dtype))
        aggregated = aggregated / degree.clamp_min(1.0)
        updated = self.update(torch.cat([node_repr, aggregated], dim=-1))
        return self.norm(node_repr + updated)

    def forward(
        self,
        node_conn: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # node_conn: [W, C, 7], edge_attr: [W, E, 4]
        if node_conn.dim() != 3:
            raise ValueError("LocalGraphEncoder expects node_conn with shape [W, C, F].")

        node_hidden = self.node_proj(node_conn)
        window_outputs: List[torch.Tensor] = []
        for window_idx in range(node_hidden.shape[0]):
            window_outputs.append(
                self._message_passing(
                    node_hidden[window_idx],
                    edge_index,
                    edge_attr[window_idx] if edge_attr.numel() > 0 else edge_attr.new_zeros((0, edge_attr.shape[-1])),
                )
            )
        graph_hidden = torch.stack(window_outputs, dim=0).mean(dim=0)
        if channel_mask is not None:
            graph_hidden = graph_hidden * channel_mask.unsqueeze(-1).float()
        return graph_hidden


class CoTARBlock(nn.Module):
    def __init__(self, d_model: int, *, num_heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.ff_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.attn_norm(x + attn_out)
        x = self.ff_norm(x + self.ff(x))
        return x


class Model(nn.Module):
    def __init__(self, args: Any) -> None:
        super().__init__()
        feature_dim = int(_get_arg(args, "feature_dim", 14))
        conn_dim = int(_get_arg(args, "conn_dim", 7))
        edge_dim = int(_get_arg(args, "edge_dim", 4))
        d_model = int(_get_arg(args, "d_model", 96))
        dropout = float(_get_arg(args, "dropout", 0.2))
        kernel_size = int(_get_arg(args, "temporal_kernel_size", 3))
        comparator_layers = int(_get_arg(args, "comparator_layers", 2))
        num_heads = int(_get_arg(args, "num_heads", 4))

        self.temporal_encoder = TemporalChannelEncoder(
            feature_dim,
            d_model,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.graph_encoder = LocalGraphEncoder(conn_dim, edge_dim, d_model, dropout=dropout)
        self.run_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.run_channel_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.patient_comparator = nn.ModuleList(
            [CoTARBlock(d_model, num_heads=num_heads, dropout=dropout) for _ in range(comparator_layers)]
        )
        self.patient_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def _fuse_runs(self, run_outputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        patient_channels = run_outputs[0]["run_emb"].shape[0]
        emb_dim = run_outputs[0]["run_emb"].shape[-1]
        fused_emb = run_outputs[0]["run_emb"].new_zeros((patient_channels, emb_dim))
        fused_logits = run_outputs[0]["run_logits"].new_zeros(patient_channels)
        denom = run_outputs[0]["run_logits"].new_zeros(patient_channels)

        for run_output in run_outputs:
            channel_mask = run_output["channel_mask"].float()
            run_weight = float(run_output["quality_weight"])
            weight = channel_mask * run_weight
            fused_emb += run_output["run_emb"] * weight.unsqueeze(-1)
            fused_logits += run_output["run_logits"] * weight
            denom += weight

        patient_mask = denom > 0
        denom = denom.clamp_min(1.0)
        patient_emb = fused_emb / denom.unsqueeze(-1)
        prior_logits = fused_logits / denom
        return {
            "patient_emb": patient_emb,
            "prior_logits": prior_logits,
            "patient_mask": patient_mask,
        }

    def forward(self, patient_batch: Dict[str, Any]) -> Dict[str, Any]:
        run_outputs: List[Dict[str, torch.Tensor]] = []
        for run in patient_batch["runs"]:
            temporal_out = self.temporal_encoder(run["x_feat"], run["channel_mask"])
            graph_emb = self.graph_encoder(
                run["node_conn"],
                run["edge_index"],
                run["edge_attr"],
                run["channel_mask"],
            )
            run_emb = self.run_fusion(torch.cat([temporal_out["channel_emb"], graph_emb], dim=-1))
            run_logits = self.run_channel_head(run_emb).squeeze(-1)
            run_logits = run_logits * run["channel_mask"].float()
            run_outputs.append(
                {
                    "run_id": run["run_id"],
                    "quality_weight": run["quality_weight"],
                    "channel_mask": run["channel_mask"],
                    "run_emb": run_emb,
                    "run_logits": run_logits,
                    "window_logits": temporal_out["window_logits"],
                }
            )

        fused = self._fuse_runs(run_outputs)
        patient_tokens = fused["patient_emb"].unsqueeze(0)
        key_padding_mask = (~fused["patient_mask"]).unsqueeze(0)
        for block in self.patient_comparator:
            patient_tokens = block(patient_tokens, key_padding_mask=key_padding_mask)

        patient_logits = self.patient_head(patient_tokens.squeeze(0)).squeeze(-1)
        patient_logits = patient_logits + 0.5 * fused["prior_logits"]
        patient_logits = patient_logits.masked_fill(~fused["patient_mask"], -12.0)
        patient_scores = torch.sigmoid(patient_logits) * fused["patient_mask"].float()

        run_channel_scores = [
            torch.sigmoid(run_output["run_logits"]) * run_output["channel_mask"].float()
            for run_output in run_outputs
        ]

        return {
            "patient_logits": patient_logits,
            "patient_scores": patient_scores,
            "patient_mask": fused["patient_mask"],
            "run_outputs": run_outputs,
            "run_channel_scores": run_channel_scores,
        }


TeChEZModel = Model

__all__ = ["Model", "TeChEZModel"]
