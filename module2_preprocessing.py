import re

import mne
import numpy as np
import pandas as pd


def normalize_channel_name(name: str) -> str:
    if not isinstance(name, str):
        return str(name)

    channel_name = name.upper().strip()
    if channel_name.startswith("EEG "):
        channel_name = channel_name[4:].strip()
    elif channel_name.startswith("POL "):
        channel_name = channel_name[4:].strip()

    for suffix in ["-REF", " REF", "_REF"]:
        if channel_name.endswith(suffix):
            channel_name = channel_name[:-len(suffix)].strip()

    channel_name = channel_name.replace(" ", "")
    match = re.match(r"^([A-Z]+)0*(\d+)$", channel_name)
    if match:
        channel_name = f"{match.group(1)}{match.group(2)}"
    return channel_name


def load_and_preprocess_edf(
    edf_path,
    channels_path,
    target_sfreq=512.0,
    bandpass_low=1.0,
    bandpass_high=250.0,
):
    if float(target_sfreq) < float(bandpass_high) * 2.0:
        raise ValueError(
            f"target_sfreq={target_sfreq} Hz is too low for a {bandpass_high} Hz spectral ceiling."
        )

    channels_df = pd.read_csv(channels_path, sep="\t")
    channels_df["name_norm"] = channels_df["name"].apply(normalize_channel_name)
    channels_df["type_upper"] = channels_df["type"].astype(str).str.upper()

    valid_mask = channels_df["type_upper"].isin(["ECOG", "SEEG", "STEREOEEG"]) & (channels_df["status"] != "bad")
    valid_channels_norm = channels_df.loc[valid_mask, "name_norm"].tolist()

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

    original_to_norm = {}
    norm_to_original = {}
    for channel_name in raw.ch_names:
        normalized = normalize_channel_name(channel_name)
        original_to_norm[channel_name] = normalized
        if normalized not in norm_to_original:
            norm_to_original[normalized] = channel_name

    raw.rename_channels(original_to_norm)
    existing_valid_norm = [channel for channel in valid_channels_norm if channel in raw.ch_names]

    seen = set()
    unique_existing_valid_norm = []
    for channel_name in existing_valid_norm:
        if channel_name not in seen:
            unique_existing_valid_norm.append(channel_name)
            seen.add(channel_name)

    raw.pick(unique_existing_valid_norm)

    freqs = np.arange(60, raw.info["sfreq"] / 2, 60)
    if len(freqs) > 0:
        raw.notch_filter(freqs=freqs, verbose="ERROR")

    raw_sfreq = float(raw.info["sfreq"])
    raw_nyquist = raw_sfreq / 2.0
    if raw_nyquist < float(bandpass_high):
        print(
            f"WARNING: {edf_path} original sampling rate {raw_sfreq:.2f} Hz cannot faithfully preserve "
            f"frequencies up to {bandpass_high:.1f} Hz before resampling."
        )

    h_freq = min(float(bandpass_high), raw_nyquist - 0.1)
    if h_freq <= float(bandpass_low):
        raise ValueError(
            f"Invalid bandpass for {edf_path}: low={bandpass_low} Hz, high={h_freq:.2f} Hz."
        )

    raw.filter(l_freq=float(bandpass_low), h_freq=h_freq, verbose="ERROR")
    if float(raw.info["sfreq"]) != float(target_sfreq):
        raw.resample(target_sfreq)

    data = raw.get_data().astype(np.float32)
    data = data - np.median(data, axis=0, keepdims=True)
    return raw, unique_existing_valid_norm, data, original_to_norm
