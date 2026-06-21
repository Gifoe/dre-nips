from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import math
import os
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_LZU_ROOT = Path(r"F:\兰大二院新SEEG数据")
DEFAULT_DATAEST_ROOT = Path(r"E:\DRE-nips\dataest")
DEFAULT_LZU_EZ_XLSX = DEFAULT_LZU_ROOT / "label-seeg_with_engel.xlsx"
DEFAULT_LZU_TIME_XLSX = DEFAULT_LZU_ROOT / "SEEG数据分析时间标签.xlsx"
DEFAULT_MULTICENTER_SIDECAR_ROOT = DEFAULT_DATAEST_ROOT
DEFAULT_MULTICENTER_PARTICIPANTS = DEFAULT_DATAEST_ROOT / "participants-muticenter.tsv"
DEFAULT_READ_AUDIT_DIR = Path(r"D:\nips-temp\data_audit")

INTRACRANIAL_TYPES = {"ECOG", "SEEG", "STEREOEEG", "IEEG"}
EXCLUDED_NAME_PREFIXES = ("DC", "$")


@dataclass
class SeizureRecord:
    subject_id: str
    seizure_id: str
    signal: np.ndarray
    sfreq: float
    channel_names: list[str]
    seizure_onset_sec: float
    seizure_offset_sec: float | None
    labels: np.ndarray
    channel_meta: list[dict]
    split: str | None = None


@dataclass
class PatientRecord:
    subject_id: str
    seizures: list[SeizureRecord]
    canonical_channels: list[str]
    labels: np.ndarray
    channel_meta: list[dict]


@dataclass
class DataInterfaceConfig:
    datasets: Sequence[str] = ("lzu", "hup", "multicenter")
    lzu_root: Path = DEFAULT_LZU_ROOT
    lzu_ez_annotations_path: Path = DEFAULT_LZU_EZ_XLSX
    lzu_seizure_times_path: Path = DEFAULT_LZU_TIME_XLSX
    hup_root: Path = DEFAULT_DATAEST_ROOT
    hup_participants_path: Path | None = None
    multicenter_root: Path = DEFAULT_DATAEST_ROOT
    multicenter_sidecar_root: Path | None = DEFAULT_MULTICENTER_SIDECAR_ROOT
    multicenter_participants_path: Path | None = DEFAULT_MULTICENTER_PARTICIPANTS
    read_audit_dir: Path = DEFAULT_READ_AUDIT_DIR
    success_only: bool = True
    subject_filter: Sequence[str] | str | None = None
    target_sfreq: float | None = 512.0
    bandpass_low: float | None = 1.0
    bandpass_high: float | None = 150.0
    line_freq: float | None = None
    ez_definition: str = "soz_only"
    strict: bool = False
    debug_limit: int | None = None
    feature_num_workers: int = 20
    write_read_audit: bool = True


def coerce_config(args: Any | None) -> DataInterfaceConfig:
    if args is None:
        return DataInterfaceConfig()
    if isinstance(args, DataInterfaceConfig):
        return args

    def get(name: str, default: Any) -> Any:
        if isinstance(args, Mapping):
            return args.get(name, default)
        return getattr(args, name, default)

    datasets = get("datasets", get("dataset_names", DataInterfaceConfig.datasets))
    if isinstance(datasets, str):
        datasets = [part for part in re.split(r"[,;\s]+", datasets) if part]
    datasets = tuple(str(part).strip().lower() for part in datasets)
    if "all" in datasets:
        datasets = ("lzu", "hup", "multicenter")

    return DataInterfaceConfig(
        datasets=datasets,
        lzu_root=Path(get("lzu_root", DEFAULT_LZU_ROOT)),
        lzu_ez_annotations_path=Path(get("lzu_ez_annotations_path", DEFAULT_LZU_EZ_XLSX)),
        lzu_seizure_times_path=Path(get("lzu_seizure_times_path", DEFAULT_LZU_TIME_XLSX)),
        hup_root=Path(get("hup_root", get("dataset_dir", DEFAULT_DATAEST_ROOT))),
        hup_participants_path=optional_path(get("hup_participants_path", None)),
        multicenter_root=Path(get("multicenter_root", get("dataset_dir", DEFAULT_DATAEST_ROOT))),
        multicenter_sidecar_root=optional_path(get("multicenter_sidecar_root", DEFAULT_MULTICENTER_SIDECAR_ROOT)),
        multicenter_participants_path=optional_path(
            get("multicenter_participants_path", DEFAULT_MULTICENTER_PARTICIPANTS)
        ),
        read_audit_dir=Path(get("read_audit_dir", DEFAULT_READ_AUDIT_DIR)),
        subject_filter=get("subject_filter", None),
        success_only=bool(get("success_only", True)),
        target_sfreq=optional_float(get("target_sfreq", 512.0)),
        bandpass_low=optional_float(get("bandpass_low", 1.0)),
        bandpass_high=optional_float(get("bandpass_high", 150.0)),
        line_freq=optional_float(get("line_freq", None)),
        ez_definition=str(get("ez_definition", "soz_only")).lower(),
        strict=bool(get("strict", False)),
        debug_limit=get("debug_limit", None),
        feature_num_workers=int(get("feature_num_workers", 20)),
        write_read_audit=bool(get("write_read_audit", True)),
    )


def optional_path(value: Any) -> Path | None:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return Path(value)


def optional_float(value: Any) -> float | None:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def normalize_outcome_text(value: Any) -> str:
    text = clean_text(value)
    text = (
        text.replace("\u2160", "I")
        .replace("\u2161", "II")
        .replace("\u2162", "III")
        .replace("\u2163", "IV")
        .replace("\uff11", "1")
        .replace("\uff12", "2")
        .replace("\uff13", "3")
        .replace("\uff14", "4")
    )
    return re.sub(r"\s+", "", text).upper()


def is_successful_surgery_value(value: Any) -> bool | None:
    text = normalize_outcome_text(value)
    if not text:
        return None
    if text in {"S", "SUCCESS", "SUCCESSFUL", "TRUE", "YES", "Y", "1"}:
        return True
    if text in {"F", "FAIL", "FAILED", "FALSE", "NO", "N", "0", "NR"}:
        return False
    if text in {"I", "I级", "ENGELI", "ENGELI级", "ENGEL1", "ENGEL1级"}:
        return True
    if text.startswith("ENGELII") or text.startswith("ENGELIII") or text.startswith("ENGELIV"):
        return False
    if text in {"II", "II级", "III", "III级", "IV", "IV级"}:
        return False
    return None


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def normalize_channel_name(name: Any) -> str:
    text = clean_text(name).upper()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[-_.]+", "", text)
    text = text.replace("EEG", "")
    match = re.fullmatch(r"([A-Z]+)0*([0-9]+)", text)
    if match:
        return f"{match.group(1)}{int(match.group(2))}"
    return text


def natural_channel_sort_key(name: Any) -> tuple:
    norm = normalize_channel_name(name)
    parts = re.split(r"(\d+)", norm)
    key: list[Any] = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part)
    return tuple(key)


def parse_contact_topology(channel_name_norm: str) -> tuple[str, int | None]:
    match = re.match(r"^([A-Z]+)(\d+)$", channel_name_norm)
    if match:
        return match.group(1), int(match.group(2))
    return channel_name_norm, None


def make_unique(names: Sequence[str]) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    unique: list[str] = []
    for name in names:
        counts[name] += 1
        unique.append(name if counts[name] == 1 else f"{name}__DUP{counts[name]}")
    return unique


def subject_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean_text(value).lower())


def subject_filter_set(value: Sequence[str] | str | None, add_sub_prefix: bool) -> set[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        raw_items = [part for part in re.split(r"[,;\s]+", value) if part]
    else:
        raw_items = [str(part) for part in value if str(part).strip()]
    result: set[str] = set()
    for item in raw_items:
        result.add(item)
        result.add(subject_key(item))
        if add_sub_prefix and not item.startswith("sub-"):
            result.add(f"sub-{item}")
            result.add(subject_key(f"sub-{item}"))
    return result


def as_binary(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))
    try:
        numeric = float(value)
        if math.isfinite(numeric):
            if numeric == 1.0:
                return 1
            if numeric == 0.0:
                return 0
            return None
    except (TypeError, ValueError):
        pass
    text = clean_text(value).lower()
    if text in {"true", "yes", "y", "t", "soz", "ez", "resected", "good"}:
        return 1
    if text in {"false", "no", "n", "f", "bad", "0"}:
        return 0
    return None


def is_excluded_channel_name(name: Any) -> bool:
    norm = normalize_channel_name(name)
    upper = clean_text(name).upper()
    return (
        not norm
        or norm.startswith(EXCLUDED_NAME_PREFIXES)
        or "EKG" in norm
        or "ECG" in norm
        or upper.startswith("$")
    )


def resolve_cpu_workers(cfg: DataInterfaceConfig, total: int) -> int:
    requested = int(getattr(cfg, "feature_num_workers", 20))
    if requested <= 0 or total <= 1:
        return 1
    return max(1, min(requested, total, os.cpu_count() or 1))


def cpu_worker_initializer() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
    except Exception:
        pass


def build_patient_records(seizures: Iterable[SeizureRecord]) -> list[PatientRecord]:
    grouped: dict[str, list[SeizureRecord]] = defaultdict(list)
    for seizure in seizures:
        validate_seizure_shape(seizure)
        grouped[seizure.subject_id].append(seizure)

    patients: list[PatientRecord] = []
    for subject_id, subject_seizures in sorted(grouped.items()):
        meta_by_channel: dict[str, dict] = {}
        labels_by_channel: dict[str, float] = {}
        for seizure in subject_seizures:
            for idx, channel_name in enumerate(seizure.channel_names):
                meta = dict(seizure.channel_meta[idx]) if idx < len(seizure.channel_meta) else {}
                meta.setdefault("channel_name_norm", channel_name)
                group, number = parse_contact_topology(channel_name)
                meta.setdefault("contact_group", group)
                meta.setdefault("contact_number", number)
                meta_by_channel.setdefault(channel_name, meta)
                label = float(seizure.labels[idx])
                labels_by_channel[channel_name] = min(labels_by_channel.get(channel_name, 1.0), label)

        canonical_channels = sorted(meta_by_channel, key=lambda name: canonical_channel_sort_key(meta_by_channel[name]))
        labels = np.asarray([labels_by_channel[name] for name in canonical_channels], dtype=np.float32)
        patients.append(
            PatientRecord(
                subject_id=subject_id,
                seizures=subject_seizures,
                canonical_channels=canonical_channels,
                labels=labels,
                channel_meta=[meta_by_channel[name] for name in canonical_channels],
            )
        )
    return patients


def validate_seizure_shape(seizure: SeizureRecord) -> None:
    if seizure.signal.ndim != 2:
        raise ValueError(f"{seizure.subject_id}/{seizure.seizure_id}: signal must be [channels, time].")
    n_channels = seizure.signal.shape[0]
    if n_channels != len(seizure.channel_names):
        raise ValueError(f"{seizure.subject_id}/{seizure.seizure_id}: channel count mismatch.")
    if n_channels != int(seizure.labels.shape[0]):
        raise ValueError(f"{seizure.subject_id}/{seizure.seizure_id}: label count mismatch.")
    if n_channels != len(seizure.channel_meta):
        raise ValueError(f"{seizure.subject_id}/{seizure.seizure_id}: channel_meta count mismatch.")
    if seizure.seizure_onset_sec is None:
        raise ValueError(f"{seizure.subject_id}/{seizure.seizure_id}: seizure onset is missing.")


def canonical_channel_sort_key(meta: Mapping[str, Any]) -> tuple:
    if meta.get("lzu_original_channel_id") is not None:
        return ("", int(meta["lzu_original_channel_id"]), str(meta.get("channel_name_norm", "")))
    group = str(meta.get("contact_group", ""))
    number = meta.get("contact_number")
    number_key = int(number) if number is not None and not pd.isna(number) else 10**9
    return (group, number_key, str(meta.get("channel_name_norm", "")))
