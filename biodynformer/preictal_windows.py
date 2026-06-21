from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PreictalWindow:
    name: str
    start_sec: float
    end_sec: float


DEFAULT_WINDOWS: tuple[PreictalWindow, ...] = (
    PreictalWindow("P1", -120.0, -90.0),
    PreictalWindow("P2", -90.0, -60.0),
    PreictalWindow("P3", -60.0, -30.0),
    PreictalWindow("P4", -30.0, 0.0),
)


def extract_preictal_segments(
    signal: np.ndarray,
    *,
    sfreq: float,
    onset_sec: float,
    windows: Sequence[PreictalWindow] = DEFAULT_WINDOWS,
) -> tuple[list[np.ndarray], np.ndarray]:
    data = np.asarray(signal, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("signal must be [channels, samples].")
    sfreq = float(sfreq)
    onset_sec = float(onset_sec)
    segments: list[np.ndarray] = []
    mask = np.zeros((len(windows),), dtype=bool)
    num_samples = data.shape[1]
    for idx, window in enumerate(windows):
        start = int(round((onset_sec + window.start_sec) * sfreq))
        end = int(round((onset_sec + window.end_sec) * sfreq))
        if start < 0 or end > num_samples or end <= start:
            segments.append(np.zeros((data.shape[0], 0), dtype=np.float32))
            continue
        segments.append(data[:, start:end].astype(np.float32, copy=False))
        mask[idx] = True
    return segments, mask
