from __future__ import annotations

import re
from typing import Sequence

import numpy as np


NODE_FEATURE_NAMES = [
    "high_gamma_proxy",
    "line_length",
    "aperiodic_slope",
    "aperiodic_offset",
    "graph_strength",
    "network_density",
    "global_efficiency_proxy",
]
QUALITY_FEATURE_NAMES = [
    "artifact_ratio",
    "valid_duration",
    "bad_channel_mask",
    "line_noise_ratio",
    "hfo_detection_quality",
    "num_valid_windows",
]
COVERAGE_FEATURE_NAMES = [
    "shaft_size",
    "contact_index",
    "relative_contact_position",
    "same_shaft_degree",
]


def _safe_corrcoef(data: np.ndarray) -> np.ndarray:
    if data.shape[1] < 2:
        return np.zeros((data.shape[0], data.shape[0]), dtype=np.float32)
    centered = data - data.mean(axis=1, keepdims=True)
    std = centered.std(axis=1)
    valid = std > 1e-8
    corr = np.zeros((data.shape[0], data.shape[0]), dtype=np.float32)
    if np.any(valid):
        normed = centered[valid] / std[valid, None]
        corr_valid = normed.dot(normed.T) / float(max(data.shape[1], 1))
        corr[np.ix_(valid, valid)] = corr_valid.astype(np.float32)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    return corr.astype(np.float32)


def compute_node_features(segment: np.ndarray, sfreq: float) -> np.ndarray:
    data = np.asarray(segment, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("segment must be [channels, samples].")
    if data.shape[1] < 2:
        return np.zeros((data.shape[0], len(NODE_FEATURE_NAMES)), dtype=np.float32)
    duration = max(data.shape[1] / max(float(sfreq), 1e-6), 1e-6)
    diffs = np.diff(data, axis=1)
    line_length = np.sum(np.abs(diffs), axis=1) / duration
    high_gamma_proxy = np.log1p(np.mean(diffs * diffs, axis=1))

    freqs = np.fft.rfftfreq(data.shape[1], d=1.0 / max(float(sfreq), 1e-6))
    spectrum = np.abs(np.fft.rfft(data, axis=1)) ** 2
    freq_mask = freqs > 0
    x = np.log(freqs[freq_mask] + 1e-6)
    y = np.log(spectrum[:, freq_mask] + 1e-6)
    x_centered = x - x.mean()
    denom = float(np.sum(x_centered * x_centered)) or 1.0
    slopes = np.sum((y - y.mean(axis=1, keepdims=True)) * x_centered[None, :], axis=1) / denom
    offsets = y.mean(axis=1)

    corr_abs = np.abs(_safe_corrcoef(data))
    graph_strength = corr_abs.mean(axis=1)
    density = np.full((data.shape[0],), float(np.mean(corr_abs > 0.3)), dtype=np.float32)
    efficiency = graph_strength.copy()
    features = np.stack(
        [
            high_gamma_proxy,
            line_length,
            slopes,
            offsets,
            graph_strength,
            density,
            efficiency,
        ],
        axis=1,
    )
    return np.nan_to_num(features.astype(np.float32), copy=False)


def compute_quality_features(segment: np.ndarray, sfreq: float, valid_window_count: int) -> np.ndarray:
    data = np.asarray(segment, dtype=np.float32)
    channels = int(data.shape[0])
    if data.shape[1] < 2:
        return np.zeros((channels, len(QUALITY_FEATURE_NAMES)), dtype=np.float32)
    duration = data.shape[1] / max(float(sfreq), 1e-6)
    amplitude = np.abs(data)
    channel_std = np.std(data, axis=1)
    artifact_ratio = np.mean(amplitude > (np.median(amplitude, axis=1, keepdims=True) + 8.0 * channel_std[:, None] + 1e-6), axis=1)
    bad = (channel_std <= 1e-8).astype(np.float32)
    line_noise = np.mean(np.abs(np.diff(data, axis=1)), axis=1) / (np.mean(amplitude, axis=1) + 1e-6)
    hfo_quality = np.clip(1.0 - artifact_ratio - bad, 0.0, 1.0)
    out = np.stack(
        [
            artifact_ratio,
            np.full((channels,), duration, dtype=np.float32),
            bad,
            line_noise,
            hfo_quality,
            np.full((channels,), float(valid_window_count), dtype=np.float32),
        ],
        axis=1,
    )
    return np.nan_to_num(out.astype(np.float32), copy=False)


def compute_sync_edge(segment: np.ndarray) -> np.ndarray:
    return np.abs(_safe_corrcoef(np.asarray(segment, dtype=np.float32))).astype(np.float32)


def compute_causal_edge(segment: np.ndarray) -> np.ndarray:
    data = np.asarray(segment, dtype=np.float32)
    channels = data.shape[0]
    if data.shape[1] < 3:
        return np.zeros((channels, channels), dtype=np.float32)
    src_now = data[:, :-1]
    dst_next = data[:, 1:]
    edge = np.zeros((channels, channels), dtype=np.float32)
    for src in range(channels):
        src_vec = src_now[src]
        src_std = float(np.std(src_vec)) + 1e-6
        for dst in range(channels):
            if src == dst:
                continue
            dst_vec = dst_next[dst]
            dst_std = float(np.std(dst_vec)) + 1e-6
            edge[src, dst] = float(np.mean((src_vec - src_vec.mean()) * (dst_vec - dst_vec.mean())) / (src_std * dst_std))
    return np.nan_to_num(edge, nan=0.0).astype(np.float32)


def _channel_parts(name: str) -> tuple[str, int | None]:
    match = re.match(r"^([A-Za-z]+)(\d+)$", str(name).replace(" ", ""))
    if not match:
        return str(name), None
    return match.group(1).upper(), int(match.group(2))


def compute_structural_edge(channel_names: Sequence[str]) -> np.ndarray:
    channels = len(channel_names)
    parts = [_channel_parts(name) for name in channel_names]
    edge = np.zeros((channels, channels), dtype=np.float32)
    for i, (shaft_i, contact_i) in enumerate(parts):
        for j, (shaft_j, contact_j) in enumerate(parts):
            if i == j or shaft_i != shaft_j:
                continue
            if contact_i is None or contact_j is None:
                edge[i, j] = 0.5
            else:
                edge[i, j] = 1.0 / (1.0 + abs(contact_i - contact_j))
    return edge


def compute_coverage_features(channel_names: Sequence[str]) -> np.ndarray:
    parts = [_channel_parts(name) for name in channel_names]
    shaft_counts: dict[str, int] = {}
    for shaft, _ in parts:
        shaft_counts[shaft] = shaft_counts.get(shaft, 0) + 1
    rows = []
    for shaft, contact in parts:
        shaft_size = float(shaft_counts[shaft])
        idx = float(contact or 0)
        rel = idx / max(shaft_size, 1.0) if contact is not None else 0.0
        same_shaft_degree = max(shaft_size - 1.0, 0.0)
        rows.append([shaft_size, idx, rel, same_shaft_degree])
    return np.asarray(rows, dtype=np.float32)
