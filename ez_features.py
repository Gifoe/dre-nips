from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from graph_channel import (
    DEFAULT_SPECTRAL_BANDS,
    DEFAULT_SPECTRAL_MAX_FREQ,
    DEFAULT_SPECTRAL_MIN_FREQ,
    build_dynamic_graphs_batched,
    configure_runtime,
    extract_connectivity_node_features_batch,
)
from module2_preprocessing import load_and_preprocess_edf
from module3_labels_metadata import parse_channel_labels
from module4_time_windows import create_time_windows


FEATURE_DIM = 14


@dataclass
class RunFeatureRecord:
    subject_id: str
    run_id: str
    task: str
    phase_group: str
    channel_names_norm: List[str]
    contact_groups: List[str]
    contact_numbers: List[Optional[int]]
    labels: np.ndarray
    phase_ids: np.ndarray
    quality_weight: float
    x_feat: np.ndarray
    node_conn: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    n_windows: int
    sfreq: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "run_id": self.run_id,
            "task": self.task,
            "phase_group": self.phase_group,
            "channel_names_norm": list(self.channel_names_norm),
            "contact_groups": list(self.contact_groups),
            "contact_numbers": list(self.contact_numbers),
            "labels": np.asarray(self.labels, dtype=np.float32),
            "phase_ids": np.asarray(self.phase_ids, dtype=np.int64),
            "quality_weight": float(self.quality_weight),
            "x_feat": np.asarray(self.x_feat, dtype=np.float32),
            "node_conn": np.asarray(self.node_conn, dtype=np.float32),
            "edge_index": np.asarray(self.edge_index, dtype=np.int64),
            "edge_attr": np.asarray(self.edge_attr, dtype=np.float32),
            "n_windows": int(self.n_windows),
            "sfreq": float(self.sfreq),
            "metadata": dict(self.metadata),
        }


def _ensure_float_tensor(array_like: Any, device: torch.device) -> torch.Tensor:
    if isinstance(array_like, torch.Tensor):
        return array_like.to(device=device, dtype=torch.float32)
    return torch.as_tensor(array_like, device=device, dtype=torch.float32)


def _stack_window_data(data: np.ndarray, usable_windows: pd.DataFrame) -> np.ndarray:
    return np.stack(
        [
            data[:, int(row.start_sample) : int(row.end_sample)]
            for row in usable_windows.itertuples(index=False)
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _compute_band_power(
    power: torch.Tensor,
    freqs: torch.Tensor,
    min_freq: float,
    max_freq: float,
) -> Tuple[torch.Tensor, List[str]]:
    restricted_mask = (freqs >= float(min_freq)) & (freqs <= float(max_freq))
    if not torch.any(restricted_mask):
        raise ValueError(
            f"No spectral bins available between {min_freq} and {max_freq} Hz."
        )

    restricted_power = power[..., restricted_mask]
    restricted_freqs = freqs[restricted_mask]

    band_names: List[str] = []
    band_powers: List[torch.Tensor] = []
    for band_name, band_low, band_high in DEFAULT_SPECTRAL_BANDS:
        low = max(float(band_low), float(min_freq))
        high = min(float(band_high), float(max_freq))
        if high <= low:
            band_power = restricted_power.new_zeros(restricted_power.shape[:2])
        else:
            band_mask = (restricted_freqs >= low) & (restricted_freqs < high)
            if torch.any(band_mask):
                band_power = restricted_power[..., band_mask].mean(dim=-1)
            else:
                band_power = restricted_power.new_zeros(restricted_power.shape[:2])
        band_names.append(band_name)
        band_powers.append(band_power)

    return torch.stack(band_powers, dim=-1), band_names


def _estimate_aperiodic_features(
    power: torch.Tensor,
    freqs: torch.Tensor,
    min_freq: float,
    max_freq: float,
) -> torch.Tensor:
    freq_mask = (freqs >= float(min_freq)) & (freqs <= float(max_freq))
    freq_mask &= freqs > 0.0
    if not torch.any(freq_mask):
        raise ValueError(
            f"No positive spectral bins available between {min_freq} and {max_freq} Hz."
        )

    x = torch.log10(freqs[freq_mask].clamp_min(1e-6))
    y = torch.log10(power[..., freq_mask].clamp_min(1e-12))

    x_mean = x.mean()
    x_centered = x - x_mean
    denom = torch.sum(x_centered.pow(2)).clamp_min(1e-8)

    y_mean = y.mean(dim=-1, keepdim=True)
    slope = torch.sum((y - y_mean) * x_centered, dim=-1) / denom
    intercept = y_mean.squeeze(-1) - slope * x_mean
    fitted = intercept.unsqueeze(-1) + slope.unsqueeze(-1) * x
    fit_error = torch.sqrt(torch.mean((y - fitted).pow(2), dim=-1).clamp_min(1e-8))

    # chi is stored as the positive steepness proxy rather than the raw negative slope.
    chi = -slope
    return torch.stack([chi, intercept, fit_error], dim=-1)


def compute_temporal_channel_features(
    windows_data: Any,
    sfreq: float,
    *,
    min_freq: float = DEFAULT_SPECTRAL_MIN_FREQ,
    max_freq: float = DEFAULT_SPECTRAL_MAX_FREQ,
    device: Optional[str] = None,
) -> torch.Tensor:
    device_obj = configure_runtime(device)
    data_t = _ensure_float_tensor(windows_data, device_obj)
    _, _, num_samples = data_t.shape

    fft_vals = torch.fft.rfft(data_t, dim=-1)
    power = torch.abs(fft_vals).pow(2) / max(int(num_samples), 1)
    freqs = torch.fft.rfftfreq(num_samples, d=1.0 / float(sfreq)).to(device_obj)
    effective_max_freq = min(float(max_freq), float(sfreq) / 2.0)

    band_power, band_names = _compute_band_power(
        power,
        freqs,
        min_freq=float(min_freq),
        max_freq=effective_max_freq,
    )
    band_log_power = torch.log1p(band_power)

    band_index = {name: idx for idx, name in enumerate(band_names)}
    eps = 1e-8
    beta = band_power[..., band_index["beta"]]
    low_gamma = band_power[..., band_index["low_gamma"]]
    high_gamma = band_power[..., band_index["high_gamma"]]
    ripple = band_power[..., band_index["ripple"]]

    ratio_high_low_gamma = (high_gamma / (low_gamma + eps)).unsqueeze(-1)
    ratio_ripple_background = (ripple / (beta + low_gamma + eps)).unsqueeze(-1)
    rms = torch.sqrt(torch.mean(data_t.pow(2), dim=-1, keepdim=True) + 1e-8)
    line_length = torch.sum(torch.abs(torch.diff(data_t, dim=-1)), dim=-1, keepdim=True)
    aperiodic = _estimate_aperiodic_features(
        power,
        freqs,
        min_freq=max(2.0, float(min_freq)),
        max_freq=min(120.0, effective_max_freq),
    )

    features = torch.cat(
        [
            band_log_power,
            ratio_high_low_gamma,
            ratio_ripple_background,
            rms,
            line_length,
            aperiodic,
        ],
        dim=-1,
    )
    if int(features.shape[-1]) != FEATURE_DIM:
        raise RuntimeError(
            f"Expected {FEATURE_DIM} temporal features, got {int(features.shape[-1])}."
        )
    return features


def _filter_contacts_to_preprocessed_channels(
    contacts_meta: pd.DataFrame,
    picked_channels_norm: Sequence[str],
) -> pd.DataFrame:
    channel_order = {name: idx for idx, name in enumerate(picked_channels_norm)}
    contacts_meta = contacts_meta[contacts_meta["is_valid"] == 1].copy()
    contacts_meta = contacts_meta[contacts_meta["channel_name_norm"].isin(channel_order)].copy()
    if contacts_meta.empty:
        return contacts_meta

    contacts_meta["picked_order"] = contacts_meta["channel_name_norm"].map(channel_order)
    contacts_meta = contacts_meta.sort_values("picked_order").reset_index(drop=True)
    return contacts_meta


def extract_run_feature_record(
    run_info: Dict[str, Any],
    *,
    device: Optional[str] = None,
    target_sfreq: float = 512.0,
    win_len_sec: float = 15.0,
    step_sec: float = 5.0,
    ez_definition: str = "soz_or_resected",
    spectral_min_freq: float = DEFAULT_SPECTRAL_MIN_FREQ,
    spectral_max_freq: float = DEFAULT_SPECTRAL_MAX_FREQ,
) -> Optional[RunFeatureRecord]:
    raw = None
    try:
        edf_path = run_info.get("edf_path")
        channels_path = run_info.get("channels_path")
        if not edf_path or not channels_path:
            return None

        raw, picked_channels_norm, data, _ = load_and_preprocess_edf(
            edf_path,
            channels_path,
            target_sfreq=target_sfreq,
            bandpass_low=spectral_min_freq,
            bandpass_high=spectral_max_freq,
        )

        contacts_meta = parse_channel_labels(channels_path, ez_definition=ez_definition)
        contacts_meta = _filter_contacts_to_preprocessed_channels(
            contacts_meta,
            picked_channels_norm,
        )
        if contacts_meta.empty:
            return None

        reorder_idx = contacts_meta["picked_order"].to_numpy(dtype=int)
        data = data[reorder_idx]

        windows_df = create_time_windows(
            raw,
            data,
            run_info,
            win_len_sec=win_len_sec,
            step_sec=step_sec,
        )
        usable_windows = windows_df.loc[
            (~windows_df["unusable_mask"])
            & windows_df["analysis_phase"].isin(["ictal", "interictal"])
        ].reset_index(drop=True)
        if usable_windows.empty:
            return None

        windows_data = _stack_window_data(data, usable_windows)
        temporal_features = compute_temporal_channel_features(
            windows_data,
            sfreq=float(raw.info["sfreq"]),
            min_freq=spectral_min_freq,
            max_freq=spectral_max_freq,
            device=device,
        )

        (
            env_corrs,
            edge_indices,
            edge_attrs,
            topology_weight_mats,
            topology_adj,
            boundary_flags,
        ) = build_dynamic_graphs_batched(
            windows_data,
            contacts_meta,
            device=device,
        )
        node_conn = extract_connectivity_node_features_batch(
            env_corrs,
            topology_weight_mats,
            topology_adj,
            boundary_flags,
        )

        edge_index = (
            edge_indices[0].numpy().astype(np.int64, copy=False)
            if edge_indices
            else np.empty((2, 0), dtype=np.int64)
        )
        edge_attr = (
            torch.stack([attr.to(dtype=torch.float32) for attr in edge_attrs], dim=0)
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
            if edge_attrs
            else np.zeros((windows_data.shape[0], 0, 4), dtype=np.float32)
        )

        phase_ids = usable_windows["analysis_phase"].map({"interictal": 0, "ictal": 1}).to_numpy(dtype=np.int64)
        ictal_fraction = float(phase_ids.mean()) if len(phase_ids) > 0 else 0.0
        quality_weight = float(len(usable_windows)) * (1.0 + ictal_fraction)

        metadata = {
            key: value
            for key, value in dict(run_info).items()
            if key not in {"edf_path", "channels_path", "events_path", "json_path"}
        }

        return RunFeatureRecord(
            subject_id=str(run_info.get("subject_id", "unknown")),
            run_id=str(run_info.get("run_id", "unknown")),
            task=str(run_info.get("task", "unknown")),
            phase_group=str(run_info.get("phase_group", "unknown")),
            channel_names_norm=contacts_meta["channel_name_norm"].tolist(),
            contact_groups=contacts_meta["contact_group"].fillna("").astype(str).tolist(),
            contact_numbers=[
                None if pd.isna(value) else int(value)
                for value in contacts_meta["contact_number"].tolist()
            ],
            labels=contacts_meta["is_ez"].to_numpy(dtype=np.float32, copy=True),
            phase_ids=phase_ids,
            quality_weight=quality_weight,
            x_feat=temporal_features.detach().cpu().numpy().astype(np.float32, copy=False),
            node_conn=node_conn.detach().cpu().numpy().astype(np.float32, copy=False),
            edge_index=edge_index,
            edge_attr=edge_attr,
            n_windows=int(windows_data.shape[0]),
            sfreq=float(raw.info["sfreq"]),
            metadata=metadata,
        )
    finally:
        if raw is not None and hasattr(raw, "close"):
            raw.close()


__all__ = [
    "FEATURE_DIM",
    "RunFeatureRecord",
    "compute_temporal_channel_features",
    "extract_run_feature_record",
]
