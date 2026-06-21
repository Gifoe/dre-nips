from __future__ import annotations

import numpy as np


HFO_FEATURE_NAMES = [
    "hfo_rate",
    "ripple_rate",
    "fast_ripple_rate",
    "hfo_amplitude",
    "hfo_duration_proxy",
    "hfo_occupancy",
    "hfo_burst_rate",
]


def compute_hfo_features(segment: np.ndarray, sfreq: float) -> np.ndarray:
    data = np.asarray(segment, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("segment must be [channels, samples].")
    if data.shape[1] < 2:
        return np.zeros((data.shape[0], len(HFO_FEATURE_NAMES)), dtype=np.float32)
    duration = max(float(data.shape[1]) / max(float(sfreq), 1e-6), 1e-6)
    centered = data - np.median(data, axis=1, keepdims=True)
    scale = np.median(np.abs(centered), axis=1, keepdims=True) / 0.6745
    scale = np.where(scale <= 1e-6, np.std(centered, axis=1, keepdims=True) + 1e-6, scale)
    z = np.abs(centered / scale)
    hfo_mask = z > 4.0
    ripple_mask = z > 5.0
    fast_mask = z > 6.0
    hfo_count = hfo_mask.sum(axis=1).astype(np.float32)
    ripple_count = ripple_mask.sum(axis=1).astype(np.float32)
    fast_count = fast_mask.sum(axis=1).astype(np.float32)
    amplitude = np.where(hfo_mask.any(axis=1), np.max(np.abs(centered), axis=1), 0.0).astype(np.float32)
    occupancy = hfo_count / float(data.shape[1])
    burst_rate = np.maximum(0.0, np.diff(hfo_mask.astype(np.int8), axis=1) > 0).sum(axis=1).astype(np.float32) / duration
    features = np.stack(
        [
            hfo_count / duration,
            ripple_count / duration,
            fast_count / duration,
            amplitude,
            occupancy * duration,
            occupancy,
            burst_rate,
        ],
        axis=1,
    )
    return np.nan_to_num(features.astype(np.float32), copy=False)
