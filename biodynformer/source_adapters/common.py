from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class SeizureRecord:
    subject_id: str
    run_id: str
    seizure_id: str
    signal: np.ndarray
    sfreq: float
    seizure_onset_sec: float
    channel_names: list[str]
    labels_ez: np.ndarray
    quality_rating: str
    channel_meta: list[dict[str, Any]]


@dataclass
class PatientRecord:
    center: str
    subject_id: str
    outcome_success: bool | None
    seizures: list[SeizureRecord]
    canonical_channels: list[str]
    labels_ez: np.ndarray
    channel_meta: list[dict[str, Any]]


def parse_bool_outcome(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower().replace("级", "")
    text = re.sub(r"\s+", "", text)
    if text in {"1", "true", "yes", "y", "s", "success", "successful", "seizurefree", "engeli", "engeli", "engel1", "i"}:
        return True
    if text in {"0", "false", "no", "n", "f", "fail", "failed", "failure", "nr", "engelii", "engeliii", "engeliv", "engel2", "engel3", "engel4", "ii", "iii", "iv"}:
        return False
    return None


def split_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in re.split(r"[,;|\s]+", str(value)) if part.strip()]


def parse_float_vector(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    return np.asarray([float(part) for part in split_list(value)], dtype=np.float32)


def normalize_channel_name(name: Any) -> str:
    text = str(name).strip().upper().replace("EEG", "")
    text = re.sub(r"[-_.\s]+", "", text)
    match = re.fullmatch(r"([A-Z]+)0*(\d+)", text)
    if match:
        return f"{match.group(1)}{int(match.group(2))}"
    return text


def read_signal(path: str | Path) -> tuple[np.ndarray, float | None, list[str] | None]:
    signal_path = Path(path)
    suffix = signal_path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(signal_path)
        return np.asarray(arr, dtype=np.float32), None, None
    if suffix == ".npz":
        payload = np.load(signal_path, allow_pickle=True)
        key = next((name for name in ("signal", "data", "x") if name in payload.files), payload.files[0])
        sfreq = float(payload["sfreq"]) if "sfreq" in payload.files else None
        channels = [str(x) for x in payload["channel_names"].tolist()] if "channel_names" in payload.files else None
        return np.asarray(payload[key], dtype=np.float32), sfreq, channels
    if suffix in {".csv", ".txt"}:
        return np.loadtxt(signal_path, delimiter="," if suffix == ".csv" else None, dtype=np.float32), None, None
    if suffix == ".edf":
        try:
            import mne  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Reading EDF requires mne. Install mne or provide .npy/.npz signal manifests.") from exc
        raw = mne.io.read_raw_edf(str(signal_path), preload=True, verbose="ERROR")
        return raw.get_data().astype(np.float32), float(raw.info["sfreq"]), [str(ch) for ch in raw.ch_names]
    raise ValueError(f"Unsupported signal file type: {signal_path}")


def resolve_manifest(center: str, manifest_path: str | Path | None, root: str | Path | None) -> Path:
    if manifest_path:
        path = Path(manifest_path)
    elif root:
        path = Path(root) / "manifest.csv"
    else:
        raise FileNotFoundError(f"No manifest path/root provided for center {center}.")
    if not path.exists():
        raise FileNotFoundError(f"{center} manifest not found: {path}")
    return path


def row_value(row: Mapping[str, Any], names: Sequence[str], default: Any = "") -> Any:
    lowered = {str(k).strip().lower(): k for k in row}
    for name in names:
        key = lowered.get(str(name).strip().lower())
        if key is not None:
            value = row.get(key)
            if value not in {None, ""}:
                return value
    return default


def load_manifest_records(center: str, manifest_path: str | Path | None = None, root: str | Path | None = None) -> list[PatientRecord]:
    manifest = resolve_manifest(center, manifest_path, root)
    base = manifest.parent
    by_subject: dict[str, dict[str, Any]] = {}
    with open(manifest, "r", encoding="utf-8-sig", newline="") as fin:
        for row in csv.DictReader(fin):
            subject_id = str(row_value(row, ("subject_id", "patient_id", "participant_id"))).strip()
            if not subject_id:
                raise ValueError(f"{manifest}: row missing subject_id.")
            signal_path = Path(str(row_value(row, ("signal_path", "edf_path", "npy_path", "npz_path"))))
            if not signal_path.is_absolute():
                signal_path = base / signal_path
            signal, sfreq_from_file, channels_from_file = read_signal(signal_path)
            sfreq = float(row_value(row, ("sfreq", "sampling_rate", "sample_rate"), sfreq_from_file))
            onset = float(row_value(row, ("seizure_onset_sec", "onset_sec", "onset"), 0.0))
            channel_names = [normalize_channel_name(ch) for ch in (split_list(row_value(row, ("channel_names", "channels"), "")) or channels_from_file or [])]
            if not channel_names:
                channel_names = [f"CH{i + 1}" for i in range(signal.shape[0])]
            labels = parse_float_vector(row_value(row, ("labels_ez", "ez_labels", "labels"), ""))
            if labels.shape[0] != signal.shape[0]:
                raise ValueError(f"{manifest}: labels_ez length {labels.shape[0]} does not match signal channels {signal.shape[0]} for {subject_id}.")
            if signal.shape[0] != len(channel_names):
                raise ValueError(f"{manifest}: channel_names length {len(channel_names)} does not match signal channels {signal.shape[0]} for {subject_id}.")
            outcome = parse_bool_outcome(row_value(row, ("outcome_success", "surgery_success", "outcome", "engel", "success"), None))
            run_id = str(row_value(row, ("run_id", "seizure_id", "file_id"), signal_path.stem)).strip()
            seizure_id = str(row_value(row, ("seizure_id", "run_id", "file_id"), run_id)).strip()
            quality = str(row_value(row, ("quality_rating", "quality", "rating"), "GOOD")).strip().upper()
            meta = {"center": center, "outcome": row_value(row, ("outcome", "engel", "success"), ""), "success_used": outcome}
            seizure = SeizureRecord(
                subject_id=subject_id,
                run_id=run_id,
                seizure_id=seizure_id,
                signal=signal,
                sfreq=sfreq,
                seizure_onset_sec=onset,
                channel_names=channel_names,
                labels_ez=labels,
                quality_rating=quality,
                channel_meta=[dict(meta) for _ in channel_names],
            )
            entry = by_subject.setdefault(
                subject_id,
                {"outcome": outcome, "seizures": [], "channels": channel_names, "labels": labels, "meta": [dict(meta) for _ in channel_names]},
            )
            entry["seizures"].append(seizure)
    return [
        PatientRecord(
            center=center,
            subject_id=subject_id,
            outcome_success=entry["outcome"],
            seizures=entry["seizures"],
            canonical_channels=entry["channels"],
            labels_ez=entry["labels"],
            channel_meta=entry["meta"],
        )
        for subject_id, entry in by_subject.items()
    ]
