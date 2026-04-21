from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


DEFAULT_SPECTRAL_MIN_FREQ = 1.0
DEFAULT_SPECTRAL_MAX_FREQ = 250.0
DEFAULT_SPECTRAL_BANDS: Sequence[Tuple[str, float, float]] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("low_gamma", 30.0, 80.0),
    ("high_gamma", 80.0, 150.0),
    ("ripple", 150.0, 250.0),
)


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    if preferred is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preferred_str = str(preferred).strip().lower()
    if preferred_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preferred_str)


def configure_runtime(preferred: Optional[str] = None) -> torch.device:
    device = resolve_device(preferred)
    torch.set_float32_matmul_precision("high")

    if device.type == "cuda":
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
    return device


def _ensure_float_tensor(array_like: Any, device: torch.device) -> torch.Tensor:
    if isinstance(array_like, torch.Tensor):
        return array_like.to(device=device, dtype=torch.float32)
    return torch.as_tensor(array_like, dtype=torch.float32, device=device)


def _build_topology_templates(contacts_meta: pd.DataFrame, device: torch.device) -> Dict[str, torch.Tensor]:
    groups = contacts_meta["contact_group"].fillna("").astype(str).to_numpy()
    numbers = pd.to_numeric(contacts_meta["contact_number"], errors="coerce").to_numpy(dtype=np.float32)

    same_group_mat = groups[:, None] == groups[None, :]
    dist_mat = np.abs(numbers[:, None] - numbers[None, :])
    dist_mat[~np.isfinite(dist_mat)] = np.inf
    dist_mat[~same_group_mat] = np.inf

    topo_adj_mat = same_group_mat & np.isfinite(dist_mat) & (dist_mat == 1.0)
    topo_adj_t = torch.as_tensor(topo_adj_mat, dtype=torch.bool, device=device)

    topo_src, topo_dst = np.where(topo_adj_mat)
    topo_src_t = torch.as_tensor(topo_src, dtype=torch.long, device=device)
    topo_dst_t = torch.as_tensor(topo_dst, dtype=torch.long, device=device)

    degree = topo_adj_t.to(dtype=torch.float32).sum(dim=-1, keepdim=True)
    boundary_flags = (degree <= 1.0).to(dtype=torch.float32)
    return {
        "topo_adj": topo_adj_t,
        "topo_src": topo_src_t,
        "topo_dst": topo_dst_t,
        "degree": degree,
        "boundary_flags": boundary_flags,
    }


def compute_envelope_correlation_batched(data_t: Any) -> torch.Tensor:
    if not isinstance(data_t, torch.Tensor):
        data_t = torch.as_tensor(data_t, dtype=torch.float32)

    data_t = data_t.to(dtype=torch.float32)
    batch_size, num_channels, num_samples = data_t.shape
    if num_channels == 0:
        return data_t.new_zeros((batch_size, 0, 0))

    fft_vals = torch.fft.fft(data_t, dim=-1)
    hilbert_kernel = torch.zeros(num_samples, dtype=torch.float32, device=data_t.device)
    if num_samples % 2 == 0:
        hilbert_kernel[0] = 1.0
        hilbert_kernel[num_samples // 2] = 1.0
        hilbert_kernel[1 : num_samples // 2] = 2.0
    else:
        hilbert_kernel[0] = 1.0
        hilbert_kernel[1 : (num_samples + 1) // 2] = 2.0

    analytic_signal = torch.fft.ifft(fft_vals * hilbert_kernel, dim=-1)
    envelope = torch.abs(analytic_signal)

    centered = envelope - envelope.mean(dim=-1, keepdim=True)
    cov = torch.bmm(centered, centered.transpose(1, 2))
    scale = torch.sqrt(torch.sum(centered.pow(2), dim=-1, keepdim=True)).clamp_min(1e-8)
    corr = cov / torch.bmm(scale, scale.transpose(1, 2)).clamp_min(1e-8)
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    eye = torch.eye(num_channels, dtype=torch.bool, device=data_t.device).unsqueeze(0)
    corr = corr.masked_fill(eye, 1.0)
    return corr


def _compute_dynamic_topology_edge_features(
    data_t: torch.Tensor,
    env_corrs: torch.Tensor,
    topo_src_t: torch.Tensor,
    topo_dst_t: torch.Tensor,
    num_channels: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = data_t.shape[0]
    if topo_src_t.numel() == 0:
        empty_edge_attr = data_t.new_zeros((batch_size, 0, 4))
        empty_weight_mat = data_t.new_zeros((batch_size, num_channels, num_channels))
        return empty_edge_attr, empty_weight_mat

    env_similarity = env_corrs[:, topo_src_t, topo_dst_t].clamp(-1.0, 1.0)

    rms = torch.sqrt(torch.mean(data_t.pow(2), dim=-1) + 1e-8)
    amp_src = rms[:, topo_src_t]
    amp_dst = rms[:, topo_dst_t]
    amp_similarity = 1.0 - torch.abs(amp_src - amp_dst) / (amp_src + amp_dst + 1e-6)
    amp_similarity = amp_similarity.clamp(0.0, 1.0)

    if data_t.size(-1) > 1:
        diff_t = torch.diff(data_t, dim=-1)
        diff_src = diff_t[:, topo_src_t, :]
        diff_dst = diff_t[:, topo_dst_t, :]
        diff_num = torch.mean(diff_src * diff_dst, dim=-1)
        diff_den = torch.sqrt(
            torch.mean(diff_src.pow(2), dim=-1) * torch.mean(diff_dst.pow(2), dim=-1)
        ).clamp_min(1e-8)
        diff_consistency = (diff_num / diff_den).clamp(-1.0, 1.0)
    else:
        diff_consistency = torch.zeros_like(env_similarity)

    amp_similarity_signed = amp_similarity * 2.0 - 1.0
    topo_weight = (0.5 * env_similarity + 0.3 * amp_similarity_signed + 0.2 * diff_consistency).clamp(-1.0, 1.0)

    edge_attr = torch.stack([topo_weight, env_similarity, amp_similarity, diff_consistency], dim=-1)
    topo_weight_mats = data_t.new_zeros((batch_size, num_channels, num_channels))
    topo_weight_mats[:, topo_src_t, topo_dst_t] = topo_weight
    return edge_attr, topo_weight_mats


def extract_connectivity_node_features_batch(
    env_corrs: Any,
    topology_weight_mats: Any,
    topology_adj: Any,
    boundary_flags: Any,
) -> torch.Tensor:
    if not isinstance(env_corrs, torch.Tensor):
        env_corrs = torch.as_tensor(env_corrs, dtype=torch.float32)
    env_corrs = env_corrs.to(dtype=torch.float32)

    topo_weight_t = _ensure_float_tensor(topology_weight_mats, env_corrs.device)
    topo_adj_t = torch.as_tensor(topology_adj, dtype=torch.bool, device=env_corrs.device)
    boundary_t = _ensure_float_tensor(boundary_flags, env_corrs.device)

    batch_size, num_channels, _ = env_corrs.shape
    if num_channels == 0:
        return env_corrs.new_zeros((batch_size, 0, 7))

    if boundary_t.dim() == 1:
        boundary_t = boundary_t.view(-1, 1)
    boundary_t = boundary_t[:num_channels]

    topo_adj_f = topo_adj_t.to(dtype=env_corrs.dtype)
    degree = topo_adj_f.sum(dim=-1, keepdim=True)
    norm_degree = degree / degree.max().clamp_min(1.0)

    degree_rep = degree.unsqueeze(0).expand(batch_size, -1, -1)
    norm_degree_rep = norm_degree.unsqueeze(0).expand(batch_size, -1, -1)
    boundary_rep = boundary_t.unsqueeze(0).expand(batch_size, -1, -1)

    strength = (topo_weight_t.abs() * topo_adj_f.unsqueeze(0)).sum(dim=-1, keepdim=True)
    neighbor_strength_sum = torch.matmul(
        topo_adj_f.unsqueeze(0).expand(batch_size, -1, -1),
        strength,
    )
    mean_neighbor_strength = neighbor_strength_sum / degree.view(1, num_channels, 1).clamp_min(1.0)

    local_efficiency = env_corrs.new_zeros((batch_size, num_channels, 1))
    clustering_coeff = env_corrs.new_zeros((batch_size, num_channels, 1))

    for node_idx in range(num_channels):
        neighbor_idx = torch.where(topo_adj_t[node_idx])[0]
        n_neighbors = int(neighbor_idx.numel())
        if n_neighbors < 2:
            continue

        sub_corr = env_corrs[:, neighbor_idx][:, :, neighbor_idx].abs()
        diag_mask = torch.eye(n_neighbors, dtype=torch.bool, device=env_corrs.device).unsqueeze(0)
        pair_sum = sub_corr.masked_fill(diag_mask, 0.0).sum(dim=(1, 2))
        norm = float(n_neighbors * (n_neighbors - 1))
        local_efficiency[:, node_idx, 0] = pair_sum / max(norm, 1.0)

        neighbor_topo = topo_adj_t[neighbor_idx][:, neighbor_idx]
        possible_edges = float(neighbor_topo.sum().item())
        if possible_edges > 0:
            sub_weights = topo_weight_t[:, neighbor_idx][:, :, neighbor_idx].abs()
            clustering_coeff[:, node_idx, 0] = (
                sub_weights * neighbor_topo.to(dtype=sub_weights.dtype).unsqueeze(0)
            ).sum(dim=(1, 2)) / possible_edges
        else:
            clustering_coeff[:, node_idx, 0] = local_efficiency[:, node_idx, 0]

    return torch.cat(
        [
            degree_rep,
            norm_degree_rep,
            strength,
            local_efficiency,
            clustering_coeff,
            mean_neighbor_strength,
            boundary_rep,
        ],
        dim=-1,
    )


def build_dynamic_graphs_batched(
    windows_data: Any,
    contacts_meta: pd.DataFrame,
    device: Optional[str] = None,
):
    device_obj = configure_runtime(device)
    data_t = _ensure_float_tensor(windows_data, device_obj)
    batch_size, num_channels, _ = data_t.shape

    if len(contacts_meta) != num_channels:
        raise ValueError(
            f"contacts_meta length ({len(contacts_meta)}) does not match channel count ({num_channels})."
        )

    env_corrs = compute_envelope_correlation_batched(data_t).to(device_obj)
    topology = _build_topology_templates(contacts_meta, device_obj)

    edge_attrs_t, topology_weight_mats = _compute_dynamic_topology_edge_features(
        data_t,
        env_corrs,
        topology["topo_src"],
        topology["topo_dst"],
        num_channels,
    )

    batched_edge_indices = []
    batched_edge_attrs = []
    empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=device_obj)
    empty_edge_attr = torch.empty((0, 4), dtype=torch.float32, device=device_obj)

    for batch_idx in range(batch_size):
        if topology["topo_src"].numel() > 0:
            final_edge_index = torch.stack([topology["topo_src"], topology["topo_dst"]], dim=0)
            final_edge_attr = edge_attrs_t[batch_idx]
        else:
            final_edge_index = empty_edge_index
            final_edge_attr = empty_edge_attr

        batched_edge_indices.append(final_edge_index.cpu())
        batched_edge_attrs.append(final_edge_attr.cpu())

    return (
        env_corrs,
        batched_edge_indices,
        batched_edge_attrs,
        topology_weight_mats,
        topology["topo_adj"],
        topology["boundary_flags"],
    )
