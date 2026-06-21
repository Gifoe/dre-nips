from __future__ import annotations

import numpy as np


def _as_channel_samples(segment: np.ndarray) -> np.ndarray:
    data = np.asarray(segment, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("segment must be shaped [channels, samples].")
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


def _fft_bandpass(data: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    if data.shape[1] < 2 or low >= sfreq / 2.0:
        return np.zeros_like(data, dtype=np.float32)
    high = min(float(high), float(sfreq) / 2.0 - 1e-3)
    if high <= low:
        return np.zeros_like(data, dtype=np.float32)
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


def _welch_or_fft_psd(data: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.signal import welch  # type: ignore

        nperseg = min(max(8, int(round(float(sfreq) * 2.0))), data.shape[1])
        freqs, psd = welch(data, fs=float(sfreq), axis=1, nperseg=nperseg)
        return freqs.astype(np.float32), psd.astype(np.float32)
    except Exception:
        freqs = np.fft.rfftfreq(data.shape[1], d=1.0 / max(float(sfreq), 1e-6))
        spec = np.fft.rfft(data, axis=1)
        psd = (np.abs(spec) ** 2) / max(data.shape[1], 1)
        return freqs.astype(np.float32), psd.astype(np.float32)


def _robust_line_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x_fit = x[finite]
    y_fit = y[finite]
    if x_fit.size < 3 or float(np.std(x_fit)) < 1e-8:
        return 0.0, 0.0, 0.0, 0.0
    median = float(np.median(y_fit))
    mad = float(np.median(np.abs(y_fit - median)))
    if mad > 1e-8:
        keep = np.abs(y_fit - median) <= 4.0 * 1.4826 * mad
        if np.count_nonzero(keep) >= 3:
            x_fit = x_fit[keep]
            y_fit = y_fit[keep]
    slope, offset = np.polyfit(x_fit, y_fit, deg=1)
    pred = slope * x_fit + offset
    residual = y_fit - pred
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-8 else 0.0
    fit_error = float(np.sqrt(np.mean(residual * residual))) if residual.size else 0.0
    return float(slope), float(offset), float(r2), fit_error


def compute_aperiodic_slope_strict(
    segment: np.ndarray,
    sfreq: float,
    *,
    freq_min: float = 2.0,
    freq_max: float = 80.0,
    line_noise_hz: float | None = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = _as_channel_samples(segment)
    channels = data.shape[0]
    if data.shape[1] < 4:
        zeros = np.zeros((channels,), dtype=np.float32)
        return zeros, zeros, zeros, zeros
    freqs, psd = _welch_or_fft_psd(data, sfreq)
    high = min(float(freq_max), max(float(sfreq) / 2.0 - 1e-3, float(freq_min)))
    mask = (freqs >= float(freq_min)) & (freqs <= high)
    if line_noise_hz is not None:
        mask &= np.abs(freqs - float(line_noise_hz)) > 1.0
    if not np.any(mask):
        zeros = np.zeros((channels,), dtype=np.float32)
        return zeros, zeros, zeros, zeros
    x = np.log10(freqs[mask] + 1e-6)
    y = np.log10(psd[:, mask] + 1e-12)
    rows = [_robust_line_fit(x, y[channel]) for channel in range(channels)]
    slope, offset, r2, fit_error = [np.asarray(values, dtype=np.float32) for values in zip(*rows)]
    return (
        np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(offset, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(r2, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(fit_error, nan=0.0, posinf=0.0, neginf=0.0),
    )


def _merge_boolean_events(mask: np.ndarray, merge_gap: int) -> list[tuple[int, int]]:
    starts = np.flatnonzero(np.diff(np.concatenate([[False], mask])) == 1)
    ends = np.flatnonzero(np.diff(np.concatenate([mask, [False]])) == -1)
    events: list[tuple[int, int]] = []
    for start, end in zip(starts, ends):
        if events and start - events[-1][1] <= merge_gap:
            events[-1] = (events[-1][0], end)
        else:
            events.append((int(start), int(end)))
    return events


def _hfo_event_stats(
    envelope: np.ndarray,
    event_mask: np.ndarray,
    *,
    sfreq: float,
    band_low: float,
    min_duration_ms: float,
    min_cycles: int,
    merge_gap_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    channels, samples = envelope.shape
    duration = max(samples / max(float(sfreq), 1e-6), 1e-6)
    min_duration_samples = int(np.ceil(min_duration_ms * float(sfreq) / 1000.0))
    min_cycle_samples = int(np.ceil(float(min_cycles) * float(sfreq) / max(float(band_low), 1e-6)))
    min_samples = max(1, min_duration_samples, min_cycle_samples)
    merge_gap = max(0, int(round(merge_gap_ms * float(sfreq) / 1000.0)))
    rates = np.zeros((channels,), dtype=np.float32)
    amp_mean = np.zeros((channels,), dtype=np.float32)
    amp_max = np.zeros((channels,), dtype=np.float32)
    for channel in range(channels):
        amps = []
        for start, end in _merge_boolean_events(event_mask[channel], merge_gap):
            if end - start < min_samples:
                continue
            event_amp = envelope[channel, start:end]
            if event_amp.size == 0:
                continue
            amps.append(float(np.max(event_amp)))
        if amps:
            values = np.asarray(amps, dtype=np.float32)
            rates[channel] = float(values.size) / duration
            amp_mean[channel] = float(np.mean(values))
            amp_max[channel] = float(np.max(values))
    return rates, amp_mean, amp_max


def compute_hfo_features_strict(
    segment: np.ndarray,
    sfreq: float,
    *,
    ripple_band: tuple[float, float] = (80.0, 250.0),
    fast_ripple_band: tuple[float, float] = (250.0, 500.0),
    threshold_mad: float = 5.0,
    min_duration_ms: float = 6.0,
    min_cycles: int = 4,
    merge_gap_ms: float = 10.0,
) -> dict[str, np.ndarray]:
    data = _as_channel_samples(segment)
    channels = data.shape[0]

    def detect_band(band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        filtered = _fft_bandpass(data, sfreq, band[0], band[1])
        envelope = np.abs(_analytic_signal(filtered)).astype(np.float32)
        med = np.median(envelope, axis=1, keepdims=True)
        mad = np.median(np.abs(envelope - med), axis=1, keepdims=True)
        scale = np.where(mad <= 1e-8, np.std(envelope, axis=1, keepdims=True) + 1e-8, mad)
        threshold = med + float(threshold_mad) * 1.4826 * scale
        mask = envelope > threshold
        artifact_samples = np.mean(mask, axis=0) > 0.5
        clean_mask = mask & ~artifact_samples[None, :]
        rate, mean_amp, max_amp = _hfo_event_stats(
            envelope,
            clean_mask,
            sfreq=sfreq,
            band_low=band[0],
            min_duration_ms=min_duration_ms,
            min_cycles=min_cycles,
            merge_gap_ms=merge_gap_ms,
        )
        artifact_ratio = np.full((channels,), float(np.mean(artifact_samples)), dtype=np.float32)
        return rate, mean_amp, max_amp, artifact_ratio

    ripple_rate, ripple_mean, ripple_max, ripple_artifact = detect_band(ripple_band)
    if float(sfreq) < 1000.0:
        zeros = np.zeros((channels,), dtype=np.float32)
        fast_rate, fast_mean, fast_max = zeros, zeros.copy(), zeros.copy()
        fast_artifact = zeros.copy()
    else:
        fast_rate, fast_mean, fast_max, fast_artifact = detect_band(fast_ripple_band)
    return {
        "ripple_rate": ripple_rate,
        "ripple_amplitude_mean": ripple_mean,
        "ripple_amplitude_max": ripple_max,
        "fast_ripple_rate": fast_rate,
        "fast_ripple_amplitude_mean": fast_mean,
        "fast_ripple_amplitude_max": fast_max,
        "hfo_artifact_ratio": np.maximum(ripple_artifact, fast_artifact).astype(np.float32),
    }


def _pac_vector_length(data: np.ndarray, sfreq: float, phase_band: tuple[float, float], amp_band: tuple[float, float]) -> np.ndarray:
    if data.shape[1] < max(int(round(float(sfreq) * 0.5)), 8):
        return np.zeros((data.shape[0],), dtype=np.float32)
    phase_signal = _fft_bandpass(data, sfreq, phase_band[0], phase_band[1])
    amp_signal = _fft_bandpass(data, sfreq, amp_band[0], amp_band[1])
    phase = np.angle(_analytic_signal(phase_signal))
    amplitude = np.abs(_analytic_signal(amp_signal))
    values = np.abs(np.mean(amplitude * np.exp(1j * phase), axis=1)) / (np.mean(amplitude, axis=1) + 1e-8)
    return np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def compute_pac_features_strict(segment: np.ndarray, sfreq: float) -> dict[str, np.ndarray]:
    data = _as_channel_samples(segment)
    return {
        "pac_theta_low_gamma": _pac_vector_length(data, sfreq, (4.0, 8.0), (30.0, 80.0)),
        "pac_theta_high_gamma": _pac_vector_length(data, sfreq, (4.0, 8.0), (80.0, 150.0)),
        "pac_alpha_low_gamma": _pac_vector_length(data, sfreq, (8.0, 13.0), (30.0, 80.0)),
        "pac_alpha_high_gamma": _pac_vector_length(data, sfreq, (8.0, 13.0), (80.0, 150.0)),
    }


def compute_local_synchrony(segment: np.ndarray) -> np.ndarray:
    data = _as_channel_samples(segment)
    if data.shape[1] < 2:
        return np.zeros((data.shape[0],), dtype=np.float32)
    centered = data - data.mean(axis=1, keepdims=True)
    std = centered.std(axis=1, keepdims=True)
    valid = std[:, 0] > 1e-8
    corr = np.zeros((data.shape[0], data.shape[0]), dtype=np.float32)
    if np.any(valid):
        normed = centered[valid] / np.clip(std[valid], 1e-8, None)
        corr[np.ix_(valid, valid)] = (normed @ normed.T / max(data.shape[1], 1)).astype(np.float32)
    np.fill_diagonal(corr, 0.0)
    return np.mean(np.abs(corr), axis=1).astype(np.float32)


def compute_physics_features_strict(
    segment: np.ndarray,
    sfreq: float,
    *,
    line_noise_hz: float | None = 50.0,
) -> np.ndarray:
    slope, offset, r2, fit_error = compute_aperiodic_slope_strict(segment, sfreq, line_noise_hz=line_noise_hz)
    hfo = compute_hfo_features_strict(segment, sfreq)
    pac = compute_pac_features_strict(segment, sfreq)
    sync = compute_local_synchrony(segment)
    out = np.stack(
        [
            slope,
            offset,
            r2,
            fit_error,
            hfo["ripple_rate"],
            hfo["ripple_amplitude_mean"],
            hfo["ripple_amplitude_max"],
            hfo["fast_ripple_rate"],
            hfo["fast_ripple_amplitude_mean"],
            hfo["fast_ripple_amplitude_max"],
            hfo["hfo_artifact_ratio"],
            pac["pac_theta_low_gamma"],
            pac["pac_theta_high_gamma"],
            pac["pac_alpha_low_gamma"],
            pac["pac_alpha_high_gamma"],
            sync,
        ],
        axis=1,
    )
    return np.nan_to_num(out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
