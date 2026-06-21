from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .schemas import DataInterfaceConfig

PREICTAL_CONTEXT_SEC = 120.0


def read_raw_edf(edf_path: Path, preload: bool) -> Any:
    try:
        import mne
    except ImportError as exc:
        raise ImportError("mne is required for EDF loading. Install mne before calling load_patient_records().") from exc
    try:
        return mne.io.read_raw_edf(
            str(edf_path),
            preload=preload,
            verbose="ERROR",
        )
    except Exception as exc:
        message = str(exc)
        if "invalid byte" in message or "encoding='latin1'" in message or "annotations channel" in message:
            return mne.io.read_raw_edf(
                str(edf_path),
                preload=preload,
                encoding="latin1",
                verbose="ERROR",
            )
        raise


def finalize_raw_data(
    raw: Any,
    picked_raw_names: Sequence[str],
    final_channel_names: Sequence[str],
    cfg: DataInterfaceConfig,
) -> tuple[np.ndarray, float, list[str], float]:
    sfreq_original = float(raw.info["sfreq"])
    raw.pick(list(picked_raw_names))
    raw.rename_channels({old: new for old, new in zip(raw.ch_names, final_channel_names)})
    if not getattr(raw, "preload", False):
        raw.load_data()

    line_freq = cfg.line_freq
    if line_freq is not None and line_freq > 0:
        freqs = np.arange(float(line_freq), float(raw.info["sfreq"]) / 2.0, float(line_freq))
        if len(freqs) > 0:
            raw.notch_filter(freqs=freqs, verbose="ERROR")

    if cfg.bandpass_low is not None or cfg.bandpass_high is not None:
        l_freq = cfg.bandpass_low
        h_freq = cfg.bandpass_high
        if h_freq is not None:
            h_freq = min(float(h_freq), float(raw.info["sfreq"]) / 2.0 - 0.1)
        if h_freq is None or l_freq is None or float(h_freq) > float(l_freq):
            raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")

    if cfg.target_sfreq is not None and abs(float(raw.info["sfreq"]) - float(cfg.target_sfreq)) > 1e-6:
        raw.resample(float(cfg.target_sfreq), verbose="ERROR")

    data = raw.get_data().astype(np.float32, copy=False)
    try:
        from scipy import signal as scipy_signal

        data = scipy_signal.detrend(data, axis=-1, type="linear").astype(np.float32, copy=False)
    except Exception:
        pass
    data -= np.median(data, axis=1, keepdims=True).astype(np.float32, copy=False)
    return data, float(raw.info["sfreq"]), list(raw.ch_names), sfreq_original


def crop_raw_to_preictal_context(raw: Any, onset_sec: float, *, preictal_sec: float = PREICTAL_CONTEXT_SEC) -> float:
    raw_duration = float(raw.n_times) / float(raw.info["sfreq"])
    onset = float(onset_sec)
    crop_start = max(0.0, onset - float(preictal_sec))
    crop_stop = min(raw_duration, onset)
    if crop_stop > crop_start:
        raw.crop(tmin=crop_start, tmax=crop_stop, include_tmax=True)
    return crop_start


def close_raw(raw: Any) -> None:
    if hasattr(raw, "close"):
        raw.close()
