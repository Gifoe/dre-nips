from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


B0_FEATURE_NAMES = [
    "log_bp_delta",
    "log_bp_theta",
    "log_bp_beta",
    "log_bp_low_gamma",
    "log_bp_high_gamma",
    "rms",
    "variance",
    "line_length_per_sec",
    "spectral_entropy",
]

PHYSICS_FEATURE_NAMES = [
    "ei_slope",
    "ei_offset",
    "hfo_rate",
    "hfo_amplitude",
    "pac_theta_gamma",
    "local_synchrony",
]

CAUSAL_NODE_FEATURE_NAMES = [
    "out_strength",
    "in_strength",
    "net_driver",
    "source_score",
    "sink_score",
    "mean_delay_out",
    "mean_delay_in",
]

TOPOLOGY_FEATURE_NAMES = [
    "causal_density_mean",
    "causal_density_std",
    "source_concentration",
    "sink_concentration",
    "driver_entropy",
    "hwc_mean",
    "hwc_std",
    "hwc_slope",
]


@dataclass(frozen=True)
class WindowConfig:
    window_length_sec: float = 2.0
    window_step_sec: float = 1.0
    pre_onset_sec: float = 60.0
    post_onset_sec: float = 120.0


def window_centers(config: WindowConfig) -> np.ndarray:
    count = int(np.floor((config.pre_onset_sec + config.post_onset_sec) / config.window_step_sec)) + 1
    return (-config.pre_onset_sec + np.arange(count, dtype=np.float32) * config.window_step_sec).astype(np.float32)


def extract_onset_windows(
    signal: np.ndarray,
    *,
    sfreq: float,
    onset_sec: float,
    config: WindowConfig,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    data = np.asarray(signal, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("signal must be shaped [channels, samples].")
    centers = window_centers(config)
    length = max(int(round(config.window_length_sec * sfreq)), 1)
    windows: list[np.ndarray] = []
    mask = np.zeros((centers.shape[0],), dtype=bool)
    for idx, rel_center in enumerate(centers):
        center_sample = int(round((onset_sec + float(rel_center)) * sfreq))
        start = center_sample - length // 2
        end = start + length
        if start < 0 or end > data.shape[1]:
            windows.append(np.zeros((data.shape[0], length), dtype=np.float32))
            continue
        windows.append(data[:, start:end].astype(np.float32, copy=False))
        mask[idx] = True
    return windows, centers, mask


def _band_power(data: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    if data.shape[1] < 2:
        return np.zeros((data.shape[0],), dtype=np.float32)
    freqs = np.fft.rfftfreq(data.shape[1], d=1.0 / max(float(sfreq), 1e-6))
    spectrum = np.abs(np.fft.rfft(data, axis=1)) ** 2
    band = (freqs >= low) & (freqs < high)
    if not np.any(band):
        return np.zeros((data.shape[0],), dtype=np.float32)
    return np.mean(spectrum[:, band], axis=1).astype(np.float32)


def compute_b0_features(segment: np.ndarray, sfreq: float) -> np.ndarray:
    data = np.asarray(segment, dtype=np.float32)
    channels = data.shape[0]
    if data.shape[1] < 2:
        return np.zeros((channels, len(B0_FEATURE_NAMES)), dtype=np.float32)
    delta = np.log1p(_band_power(data, sfreq, 1.0, 4.0))
    theta = np.log1p(_band_power(data, sfreq, 4.0, 8.0))
    beta = np.log1p(_band_power(data, sfreq, 13.0, 30.0))
    low_gamma = np.log1p(_band_power(data, sfreq, 30.0, 70.0))
    high_gamma = np.log1p(_band_power(data, sfreq, 70.0, min(150.0, sfreq / 2.0)))
    rms = np.sqrt(np.mean(data * data, axis=1))
    variance = np.var(data, axis=1)
    duration = max(data.shape[1] / max(float(sfreq), 1e-6), 1e-6)
    line_length = np.sum(np.abs(np.diff(data, axis=1)), axis=1) / duration
    freqs = np.fft.rfftfreq(data.shape[1], d=1.0 / max(float(sfreq), 1e-6))
    spectrum = np.abs(np.fft.rfft(data, axis=1)) ** 2
    spectrum_sum = np.sum(spectrum, axis=1, keepdims=True) + 1e-8
    prob = spectrum / spectrum_sum
    spectral_entropy = -np.sum(prob * np.log(prob + 1e-8), axis=1) / np.log(max(freqs.shape[0], 2))
    out = np.stack(
        [delta, theta, beta, low_gamma, high_gamma, rms, variance, line_length, spectral_entropy],
        axis=1,
    )
    return np.nan_to_num(out.astype(np.float32), copy=False)


def _aperiodic_fit(data: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.fft.rfftfreq(data.shape[1], d=1.0 / max(float(sfreq), 1e-6))
    spectrum = np.abs(np.fft.rfft(data, axis=1)) ** 2
    mask = (freqs >= 2.0) & (freqs <= min(80.0, sfreq / 2.0 - 1e-3))
    if not np.any(mask):
        return np.zeros((data.shape[0],), dtype=np.float32), np.zeros((data.shape[0],), dtype=np.float32)
    x = np.log10(freqs[mask] + 1e-6)
    y = np.log10(spectrum[:, mask] + 1e-8)
    x_centered = x - np.mean(x)
    denom = float(np.sum(x_centered * x_centered)) or 1.0
    slope = np.sum((y - np.mean(y, axis=1, keepdims=True)) * x_centered[None, :], axis=1) / denom
    offset = np.mean(y, axis=1) - slope * np.mean(x)
    return slope.astype(np.float32), offset.astype(np.float32)


def _fft_bandpass(data: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    if data.shape[1] < 2 or low >= sfreq / 2.0:
        return np.zeros_like(data, dtype=np.float32)
    high = min(high, sfreq / 2.0 - 1e-3)
    spec = np.fft.rfft(data, axis=1)
    freqs = np.fft.rfftfreq(data.shape[1], d=1.0 / max(float(sfreq), 1e-6))
    keep = (freqs >= low) & (freqs <= high)
    spec[:, ~keep] = 0
    return np.fft.irfft(spec, n=data.shape[1], axis=1).astype(np.float32)


def _analytic_signal(data: np.ndarray) -> np.ndarray:
    n = data.shape[1]
    spectrum = np.fft.fft(data, axis=1)
    h = np.zeros((n,), dtype=np.float32)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[0] = 1
        h[1 : (n + 1) // 2] = 2
    return np.fft.ifft(spectrum * h[None, :], axis=1)


def _safe_corrcoef(data: np.ndarray) -> np.ndarray:
    if data.shape[1] < 2:
        return np.zeros((data.shape[0], data.shape[0]), dtype=np.float32)
    centered = data - data.mean(axis=1, keepdims=True)
    std = centered.std(axis=1)
    valid = std > 1e-8
    out = np.zeros((data.shape[0], data.shape[0]), dtype=np.float32)
    if np.any(valid):
        normed = centered[valid] / (std[valid, None] + 1e-8)
        out[np.ix_(valid, valid)] = (normed @ normed.T / max(data.shape[1], 1)).astype(np.float32)
    np.fill_diagonal(out, 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def compute_physics_features(segment: np.ndarray, sfreq: float) -> np.ndarray:
    data = np.asarray(segment, dtype=np.float32)
    channels = data.shape[0]
    if data.shape[1] < 4:
        return np.zeros((channels, len(PHYSICS_FEATURE_NAMES)), dtype=np.float32)
    ei_slope, ei_offset = _aperiodic_fit(data, sfreq)
    hfo_band = _fft_bandpass(data, sfreq, 80.0, min(250.0, sfreq / 2.0 - 1e-3))
    scale = np.median(np.abs(hfo_band - np.median(hfo_band, axis=1, keepdims=True)), axis=1) / 0.6745
    scale = np.where(scale <= 1e-6, np.std(hfo_band, axis=1) + 1e-6, scale)
    hfo_z = np.abs(hfo_band / scale[:, None])
    duration = max(data.shape[1] / max(float(sfreq), 1e-6), 1e-6)
    hfo_events = hfo_z > 4.0
    hfo_rate = np.sum(np.diff(hfo_events.astype(np.int8), axis=1) > 0, axis=1) / duration
    hfo_amplitude = np.where(np.any(hfo_events, axis=1), np.max(np.abs(hfo_band), axis=1), 0.0)
    theta = _fft_bandpass(data, sfreq, 4.0, 8.0)
    gamma = _fft_bandpass(data, sfreq, 30.0, min(80.0, sfreq / 2.0 - 1e-3))
    theta_phase = np.angle(_analytic_signal(theta))
    gamma_amp = np.abs(_analytic_signal(gamma))
    pac = np.abs(np.mean(gamma_amp * np.exp(1j * theta_phase), axis=1)) / (np.mean(gamma_amp, axis=1) + 1e-8)
    sync = np.mean(np.abs(_safe_corrcoef(data)), axis=1)
    out = np.stack([ei_slope, ei_offset, hfo_rate, hfo_amplitude, pac, sync], axis=1)
    return np.nan_to_num(out.astype(np.float32), copy=False)


def compute_tfccm_graph(segment: np.ndarray, sfreq: float, max_delay_ms: float = 80.0) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(segment, dtype=np.float32)
    channels, samples = data.shape
    if samples < 4:
        return np.zeros((channels, channels), dtype=np.float32), np.zeros((channels, channels), dtype=np.float32)
    max_lag = max(1, int(round(max_delay_ms * sfreq / 1000.0)))
    max_lag = min(max_lag, samples // 4)
    centered = data - data.mean(axis=1, keepdims=True)
    std = centered.std(axis=1) + 1e-6
    adj = np.zeros((channels, channels), dtype=np.float32)
    delay = np.zeros((channels, channels), dtype=np.float32)
    for src in range(channels):
        for dst in range(channels):
            if src == dst:
                continue
            best = 0.0
            best_lag = 0
            for lag in range(1, max_lag + 1):
                x = centered[src, :-lag]
                y = centered[dst, lag:]
                score = float(np.mean(x * y) / (std[src] * std[dst]))
                if abs(score) > abs(best):
                    best = score
                    best_lag = lag
            adj[src, dst] = max(best, 0.0)
            delay[src, dst] = best_lag / max(float(sfreq), 1e-6)
    return np.nan_to_num(adj, nan=0.0).astype(np.float32), delay.astype(np.float32)


def compute_causal_node_features(adjacency: np.ndarray, delay: np.ndarray) -> np.ndarray:
    adj = np.asarray(adjacency, dtype=np.float32)
    delay_arr = np.asarray(delay, dtype=np.float32)
    out_strength = adj.sum(axis=1)
    in_strength = adj.sum(axis=0)
    net_driver = out_strength - in_strength
    source_score = out_strength / (out_strength + in_strength + 1e-8)
    sink_score = in_strength / (out_strength + in_strength + 1e-8)
    mean_delay_out = np.sum(delay_arr * (adj > 0), axis=1) / np.maximum(np.sum(adj > 0, axis=1), 1)
    mean_delay_in = np.sum(delay_arr * (adj > 0), axis=0) / np.maximum(np.sum(adj > 0, axis=0), 1)
    out = np.stack([out_strength, in_strength, net_driver, source_score, sink_score, mean_delay_out, mean_delay_in], axis=1)
    return np.nan_to_num(out.astype(np.float32), copy=False)


def compute_topology_features(adjacencies: np.ndarray, centers: np.ndarray) -> np.ndarray:
    graphs = np.asarray(adjacencies, dtype=np.float32)
    if graphs.ndim != 3 or graphs.shape[0] == 0:
        return np.zeros((1, len(TOPOLOGY_FEATURE_NAMES)), dtype=np.float32)
    rows = []
    for graph in graphs:
        channels = graph.shape[0]
        non_diag = ~np.eye(channels, dtype=bool)
        values = graph[non_diag]
        density = float(np.mean(values > 0.05)) if values.size else 0.0
        out_strength = graph.sum(axis=1)
        in_strength = graph.sum(axis=0)
        total_out = float(out_strength.sum()) + 1e-8
        total_in = float(in_strength.sum()) + 1e-8
        source_conc = float(np.max(out_strength) / total_out) if out_strength.size else 0.0
        sink_conc = float(np.max(in_strength) / total_in) if in_strength.size else 0.0
        driver = np.maximum(out_strength - in_strength, 0.0)
        p = driver / (float(driver.sum()) + 1e-8)
        entropy = float(-np.sum(p * np.log(p + 1e-8)) / np.log(max(channels, 2)))
        hwc = float(np.std(out_strength - in_strength))
        rows.append([density, 0.0, source_conc, sink_conc, entropy, hwc, 0.0, 0.0])
    out = np.asarray(rows, dtype=np.float32)
    if out.shape[0] > 1:
        out[:, 1] = float(np.std(out[:, 0]))
        out[:, 6] = float(np.std(out[:, 5]))
        x = np.asarray(centers[: out.shape[0]], dtype=np.float32)
        if x.size > 1:
            x_centered = x - x.mean()
            denom = float(np.sum(x_centered * x_centered)) or 1.0
            out[:, 7] = float(np.sum((out[:, 5] - out[:, 5].mean()) * x_centered) / denom)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def derive_ez_labels(labels_nez: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels_nez, dtype=np.float32)
    return np.where(labels >= 0.0, 1.0 - labels, -1.0).astype(np.float32)


def label_mask(labels_nez: np.ndarray) -> np.ndarray:
    return np.asarray(labels_nez, dtype=np.float32) >= 0.0


def stack_or_empty(rows: Iterable[np.ndarray], trailing_shape: Sequence[int]) -> np.ndarray:
    values = list(rows)
    if not values:
        return np.zeros((0, *trailing_shape), dtype=np.float32)
    return np.stack(values, axis=0).astype(np.float32)
