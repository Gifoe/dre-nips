from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .schemas import normalize_channel_name, parse_contact_topology


EVENT_ENCODINGS = ("utf-8-sig", "gbk", "gb18030")
ONSET_LABELS = {"发作开始"}
OFFSET_LABELS = {"发作结束"}
CHANNEL_LABEL_SHEET = "channel_level_labels"


@dataclass(frozen=True)
class PediatricEvent:
    label: str
    timestamp: datetime
    device_id: str


@dataclass(frozen=True)
class PediatricRun:
    subject_id: str
    edf_path: Path
    csv_path: Path
    recording_start_timestamp: datetime
    onset_timestamp: datetime
    offset_timestamp: datetime
    metadata: dict[str, Any]


@dataclass
class PediatricSeizureRecord:
    subject_id: str
    seizure_id: str
    signal: np.ndarray
    sfreq: float
    channel_names: list[str]
    seizure_onset_sec: float
    seizure_offset_sec: float | None
    labels: np.ndarray
    channel_meta: list[dict[str, Any]]
    split: str | None = None


@dataclass
class PediatricPatientRecord:
    subject_id: str
    seizures: list[PediatricSeizureRecord]
    canonical_channels: list[str]
    labels: np.ndarray
    channel_meta: list[dict[str, Any]]


def read_event_csv(csv_path: Path) -> list[PediatricEvent]:
    raw = Path(csv_path).read_bytes()
    text = None
    used_encoding = None
    for encoding in EVENT_ENCODINGS:
        try:
            text = raw.decode(encoding)
            used_encoding = encoding
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("gb18030", errors="replace")
        used_encoding = "gb18030-replace"

    rows: list[PediatricEvent] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split("\t")]
        if len(parts) < 2:
            continue
        try:
            timestamp = datetime.strptime(parts[1], "%Y/%m/%d %H:%M:%S %f")
        except ValueError:
            continue
        rows.append(PediatricEvent(label=parts[0], timestamp=timestamp, device_id=parts[2] if len(parts) > 2 else ""))
    if not rows:
        raise ValueError(f"No parseable event rows in {csv_path}; attempted encodings={EVENT_ENCODINGS}, last={used_encoding}.")
    return rows


def seizure_interval_from_csv(csv_path: Path) -> tuple[datetime, datetime]:
    events = read_event_csv(csv_path)
    onset = next((event.timestamp for event in events if event.label in ONSET_LABELS), None)
    offset = next((event.timestamp for event in events if event.label in OFFSET_LABELS and (onset is None or event.timestamp > onset)), None)
    if onset is None or offset is None or offset <= onset:
        raise ValueError(f"Missing valid 发作开始/发作结束 interval in {csv_path}.")
    return onset, offset


def recording_start_from_csv(csv_path: Path) -> datetime:
    events = read_event_csv(csv_path)
    rec_start = next((event.timestamp for event in events if "REC START" in event.label.upper()), None)
    return rec_start or min(event.timestamp for event in events)


def load_subject_metadata(xlsx_path: Path | None) -> dict[str, dict[str, Any]]:
    if xlsx_path is None:
        return {}
    path = Path(xlsx_path)
    if not path.exists():
        raise FileNotFoundError(f"Pediatric metadata workbook not found: {path}")
    out: dict[str, dict[str, Any]] = {}
    sheets = pd.read_excel(path, sheet_name=None)
    for sheet_name, df in sheets.items():
        if str(sheet_name).strip().lower() == CHANNEL_LABEL_SHEET:
            continue
        id_col = next((col for col in df.columns if str(col).strip().lower() in {"subject_id", "subject id", "被试id号", "被试id", "id"}), None)
        if id_col is None:
            continue
        for _, row in df.iterrows():
            subject_id = str(row.get(id_col, "")).strip()
            if not subject_id or subject_id.lower() == "nan":
                continue
            meta = {str(col): row.get(col) for col in df.columns}
            meta["metadata_sheet"] = str(sheet_name)
            out[subject_id] = {**out.get(subject_id, {}), **meta}
    return out


def _flag_value(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return int(default)
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    if not text or text == "nan":
        return int(default)
    if text in {"true", "yes", "y", "是", "ez"}:
        return 1
    if text in {"false", "no", "n", "否", "nez"}:
        return 0
    try:
        return 1 if float(text) > 0.5 else 0
    except ValueError:
        return int(default)


def _column_by_name(columns: Sequence[Any], candidates: Sequence[str]) -> Any | None:
    normalized = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def load_pediatric_channel_labels(xlsx_path: Path | None) -> dict[str, dict[str, dict[str, Any]]]:
    if xlsx_path is None:
        return {}
    path = Path(xlsx_path)
    if not path.exists():
        raise FileNotFoundError(f"Pediatric metadata workbook not found: {path}")
    sheets = pd.read_excel(path, sheet_name=None)
    label_df = None
    for sheet_name, df in sheets.items():
        if str(sheet_name).strip().lower() == CHANNEL_LABEL_SHEET:
            label_df = df
            break
    if label_df is None:
        return {}

    subject_col = _column_by_name(label_df.columns, ("subject_id", "subject id", "被试id号", "被试id", "id"))
    channel_col = _column_by_name(label_df.columns, ("channel_name_norm", "channel", "channel_name", "通道"))
    if subject_col is None or channel_col is None:
        raise ValueError(f"{path} sheet {CHANNEL_LABEL_SHEET!r} must contain subject_id and channel_name_norm columns.")
    ez_col = _column_by_name(label_df.columns, ("model_label_ez_excluding_bad", "is_ez_clinical", "is_ez", "ez"))
    nez_col = _column_by_name(label_df.columns, ("model_label_nez_excluding_bad", "is_nez_clinical", "is_nez", "nez"))
    usable_col = _column_by_name(label_df.columns, ("usable_channel_mask", "usable", "label_mask"))
    bad_col = _column_by_name(label_df.columns, ("is_monitoring_bad", "monitoring_bad", "bad_channel"))
    label_source_col = _column_by_name(label_df.columns, ("label_source", "source_column"))
    source_desc_col = _column_by_name(label_df.columns, ("source_description", "source_desc"))

    labels: dict[str, dict[str, dict[str, Any]]] = {}
    for _, row in label_df.iterrows():
        subject_id = str(row.get(subject_col, "")).strip()
        channel_name = normalize_channel_name(str(row.get(channel_col, "")).replace("'", ""))
        if not subject_id or subject_id.lower() == "nan" or not channel_name:
            continue
        if ez_col is not None:
            is_ez = _flag_value(row.get(ez_col), default=0)
        elif nez_col is not None:
            is_ez = 1 - _flag_value(row.get(nez_col), default=1)
        else:
            raise ValueError(f"{path} sheet {CHANNEL_LABEL_SHEET!r} must contain an EZ or NEZ label column.")
        usable = _flag_value(row.get(usable_col), default=1) if usable_col is not None else 1
        is_bad = _flag_value(row.get(bad_col), default=0) if bad_col is not None else 0
        labels.setdefault(subject_id, {})[channel_name] = {
            "is_ez": int(is_ez),
            "is_nez": int(1 - int(is_ez)),
            "usable": int(usable),
            "is_monitoring_bad": int(is_bad),
            "label_source": str(row.get(label_source_col, "channel_level_labels")) if label_source_col is not None else "channel_level_labels",
            "source_description": str(row.get(source_desc_col, "")) if source_desc_col is not None else "",
        }
    return labels


def _metadata_value(meta: dict[str, Any], candidates: Sequence[str]) -> Any:
    normalized = {str(key).strip().lower(): key for key in meta}
    for name in candidates:
        key = normalized.get(str(name).strip().lower())
        if key is not None:
            return meta.get(key)
    return None


def is_successful_pediatric_subject(meta: dict[str, Any]) -> bool | None:
    value = _metadata_value(meta, ("手术结果（成功/失败）", "手术结果", "surgery_result", "outcome"))
    text = str(value if value is not None else "").strip().lower()
    if not text or text == "nan":
        return None
    if "成功" in text or text in {"s", "success", "successful", "1", "true"}:
        return True
    if "失败" in text or text in {"f", "fail", "failed", "0", "false"}:
        return False
    return None


def _normalize_resection_text(text: Any) -> str:
    value = str(text if text is not None else "").strip()
    if value.lower() == "nan":
        return ""
    value = value.replace("；", ";").replace("，", ";").replace(",", ";").replace("、", ";")
    value = value.replace("’", "'").replace("‘", "'").replace("′", "'")
    for phrase in ("完整切除", "部分切除", "所有驻点", "驻点", "见切除方案"):
        value = value.replace(phrase, "")
    return value


def expand_channel_description(description: Any, edf_channels: Sequence[str] | None = None) -> tuple[set[str], list[str]]:
    text = _normalize_resection_text(description)
    if not text:
        return set(), []
    normalized_edf = [normalize_channel_name(ch.replace("'", "")) for ch in (edf_channels or [])]
    by_group: dict[str, list[str]] = {}
    for channel in normalized_edf:
        group, _ = parse_contact_topology(channel)
        by_group.setdefault(str(group), []).append(channel)

    channels: set[str] = set()
    uncertain: list[str] = []
    for token in [part.strip() for part in re.split(r"[;\s]+", text) if part.strip()]:
        token = token.replace("'", "")
        match = re.match(r"^([A-Za-z]+)(\d+)(?:-(\d+))?$", token)
        if match:
            group = match.group(1).upper()
            start = int(match.group(2))
            end = int(match.group(3) or start)
            if end < start:
                start, end = end, start
            for number in range(start, end + 1):
                channels.add(normalize_channel_name(f"{group}{number}"))
            continue
        group_match = re.match(r"^([A-Za-z]+)$", token)
        if group_match and by_group:
            group = group_match.group(1).upper()
            expanded = by_group.get(group, [])
            if expanded:
                channels.update(expanded)
            else:
                uncertain.append(token)
            continue
        uncertain.append(token)
    return channels, uncertain


def _pediatric_label_description(meta: dict[str, Any]) -> Any:
    for candidates in (
        ("EZ channel", "ez channel", "EZ通道"),
        ("热凝的SEEG通道名称", "热凝通道", "热凝的seeg通道名称"),
        ("手术区域（脑区名称及SEEG通道名称）", "手术区域", "切除通道"),
    ):
        value = _metadata_value(meta, candidates)
        if value is not None and str(value).strip().lower() not in {"", "nan"}:
            return value
    return ""


def _pediatric_bad_channel_description(meta: dict[str, Any]) -> Any:
    return _metadata_value(meta, ("监测坏道", "坏道", "bad channels", "bad_channel"))


def discover_pediatric_runs(root_dir: Path, metadata_xlsx: Path | None = None) -> list[PediatricRun]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Pediatric root does not exist: {root}")
    metadata_by_subject = load_subject_metadata(metadata_xlsx)
    runs: list[PediatricRun] = []
    for subject_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        subject_id = subject_dir.name
        for edf_path in sorted(subject_dir.rglob("*.edf")):
            csv_path = edf_path.with_suffix(".csv")
            if not csv_path.exists():
                candidates = sorted(edf_path.parent.glob(f"{edf_path.stem}*.csv"))
                if not candidates:
                    continue
                csv_path = candidates[0]
            onset, offset = seizure_interval_from_csv(csv_path)
            recording_start = recording_start_from_csv(csv_path)
            meta = dict(metadata_by_subject.get(subject_id, {}))
            meta.setdefault("source_center", "pediatric")
            runs.append(
                PediatricRun(
                    subject_id=subject_id,
                    edf_path=edf_path,
                    csv_path=csv_path,
                    recording_start_timestamp=recording_start,
                    onset_timestamp=onset,
                    offset_timestamp=offset,
                    metadata=meta,
                )
            )
    return runs


def _read_pediatric_raw_edf(
    edf_path: Path,
    *,
    target_sfreq: float | None,
    bandpass_low: float | None,
    bandpass_high: float | None,
    line_freq: float | None,
) -> tuple[np.ndarray, float, list[str], list[str]]:
    try:
        import mne
    except ImportError as exc:
        raise ImportError("mne is required for pediatric EDF loading.") from exc

    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    except Exception as exc:
        message = str(exc)
        if "invalid byte" in message or "encoding='latin1'" in message or "annotations channel" in message:
            raw = mne.io.read_raw_edf(str(edf_path), preload=False, encoding="latin1", verbose="ERROR")
        else:
            raise
    try:
        raw_name_by_norm: dict[str, str] = {}
        for raw_name in raw.ch_names:
            norm = normalize_channel_name(raw_name.replace("'", ""))
            upper = norm.upper()
            if not norm or upper.startswith(("DC", "$", "EKG", "ECG")) or "MARK" in upper or "TRIG" in upper:
                continue
            raw_name_by_norm.setdefault(norm, raw_name)
        picked_norms = list(raw_name_by_norm.keys())
        if not picked_norms:
            raise ValueError(f"No usable SEEG-like channels found in pediatric EDF: {edf_path}")
        picked_raw_names = [raw_name_by_norm[name] for name in picked_norms]
        raw.pick(picked_raw_names)
        raw.rename_channels({old: new for old, new in zip(raw.ch_names, picked_norms)})
        raw.load_data()
        if line_freq is not None and line_freq > 0:
            freqs = np.arange(float(line_freq), float(raw.info["sfreq"]) / 2.0, float(line_freq))
            if len(freqs) > 0:
                raw.notch_filter(freqs=freqs, verbose="ERROR")
        if bandpass_low is not None or bandpass_high is not None:
            h_freq = bandpass_high
            if h_freq is not None:
                h_freq = min(float(h_freq), float(raw.info["sfreq"]) / 2.0 - 0.1)
            if h_freq is None or bandpass_low is None or float(h_freq) > float(bandpass_low):
                raw.filter(l_freq=bandpass_low, h_freq=h_freq, verbose="ERROR")
        if target_sfreq is not None and abs(float(raw.info["sfreq"]) - float(target_sfreq)) > 1e-6:
            raw.resample(float(target_sfreq), verbose="ERROR")
        data = raw.get_data().astype(np.float32, copy=False)
        try:
            from scipy import signal as scipy_signal

            data = scipy_signal.detrend(data, axis=-1, type="linear").astype(np.float32, copy=False)
        except ImportError as exc:
            raise ImportError("scipy is required for pediatric EDF detrending.") from exc
        data -= np.median(data, axis=1, keepdims=True).astype(np.float32, copy=False)
        return data, float(raw.info["sfreq"]), list(raw.ch_names), picked_raw_names
    finally:
        if hasattr(raw, "close"):
            raw.close()


def _build_pediatric_patient_records(seizures: Sequence[PediatricSeizureRecord]) -> list[PediatricPatientRecord]:
    grouped: dict[str, list[PediatricSeizureRecord]] = {}
    for seizure in seizures:
        grouped.setdefault(seizure.subject_id, []).append(seizure)
    patients: list[PediatricPatientRecord] = []
    for subject_id, subject_seizures in sorted(grouped.items()):
        meta_by_channel: dict[str, dict[str, Any]] = {}
        labels_by_channel: dict[str, float] = {}
        for seizure in subject_seizures:
            for idx, channel_name in enumerate(seizure.channel_names):
                meta_by_channel.setdefault(channel_name, dict(seizure.channel_meta[idx]))
                labels_by_channel[channel_name] = min(float(labels_by_channel.get(channel_name, 1.0)), float(seizure.labels[idx]))
        canonical = sorted(meta_by_channel, key=lambda name: (str(meta_by_channel[name].get("contact_group", "")), int(meta_by_channel[name].get("contact_number") or 10**9), name))
        patients.append(
            PediatricPatientRecord(
                subject_id=subject_id,
                seizures=subject_seizures,
                canonical_channels=canonical,
                labels=np.asarray([labels_by_channel[name] for name in canonical], dtype=np.float32),
                channel_meta=[meta_by_channel[name] for name in canonical],
            )
        )
    return patients


def load_pediatric_patient_records(
    root_dir: Path,
    metadata_xlsx: Path | None,
    *,
    success_only: bool = True,
    subject_filter: Sequence[str] | str | None = None,
    target_sfreq: float | None = 512.0,
    bandpass_low: float | None = 1.0,
    bandpass_high: float | None = 150.0,
    line_freq: float | None = None,
    debug_limit: int | None = None,
) -> list[PediatricPatientRecord]:
    runs = discover_pediatric_runs(root_dir, metadata_xlsx)
    channel_labels_by_subject = load_pediatric_channel_labels(metadata_xlsx)
    if subject_filter:
        if isinstance(subject_filter, str):
            allowed = {part.strip() for part in re.split(r"[,;\s]+", subject_filter) if part.strip()}
        else:
            allowed = {str(part).strip() for part in subject_filter if str(part).strip()}
        runs = [run for run in runs if run.subject_id in allowed]
    if debug_limit is not None:
        runs = runs[: int(debug_limit)]

    seizures: list[PediatricSeizureRecord] = []
    for run in runs:
        success = is_successful_pediatric_subject(run.metadata)
        if success_only and success is not True:
            continue
        data, sfreq, channel_names, raw_channel_names = _read_pediatric_raw_edf(
            run.edf_path,
            target_sfreq=target_sfreq,
            bandpass_low=bandpass_low,
            bandpass_high=bandpass_high,
            line_freq=line_freq,
        )
        label_desc = _pediatric_label_description(run.metadata)
        bad_desc = _pediatric_bad_channel_description(run.metadata)
        available = set(channel_names)
        subject_channel_labels = channel_labels_by_subject.get(run.subject_id, {})
        if subject_channel_labels:
            labeled_channels = set(subject_channel_labels)
            unlabeled_edf_channels = sorted(available - labeled_channels)
            label_channels_not_in_edf = sorted(labeled_channels - available)
            matched_bad = {
                name
                for name in channel_names
                if name in subject_channel_labels and not bool(subject_channel_labels[name].get("usable", 1))
            }
            keep_indices = [
                idx
                for idx, name in enumerate(channel_names)
                if name in subject_channel_labels and bool(subject_channel_labels[name].get("usable", 1))
            ]
            matched_ez = {
                name
                for name in channel_names
                if name in subject_channel_labels
                and bool(subject_channel_labels[name].get("usable", 1))
                and int(subject_channel_labels[name].get("is_ez", 0)) == 1
            }
            unmatched_ez: list[str] = []
            uncertain_ez: list[str] = []
            uncertain_bad: list[str] = []
            label_mode = "channel_level_xlsx"
        else:
            candidate_ez, uncertain_ez = expand_channel_description(label_desc, channel_names)
            candidate_bad, uncertain_bad = expand_channel_description(bad_desc, channel_names)
            matched_ez = candidate_ez & available
            unmatched_ez = sorted(candidate_ez - available)
            matched_bad = candidate_bad & available
            keep_indices = [idx for idx, name in enumerate(channel_names) if name not in matched_bad]
            unlabeled_edf_channels = []
            label_channels_not_in_edf = []
            label_mode = "description_fallback"
        if not keep_indices:
            continue
        kept_channel_names = [channel_names[idx] for idx in keep_indices]
        kept_data = data[keep_indices]
        if subject_channel_labels:
            labels = np.asarray([0.0 if int(subject_channel_labels[name].get("is_ez", 0)) == 1 else 1.0 for name in kept_channel_names], dtype=np.float32)
        else:
            labels = np.asarray([0.0 if name in matched_ez else 1.0 for name in kept_channel_names], dtype=np.float32)
        channel_meta: list[dict[str, Any]] = []
        for out_idx, source_idx in enumerate(keep_indices):
            name = channel_names[source_idx]
            group, number = parse_contact_topology(name)
            xlsx_label = subject_channel_labels.get(name, {}) if subject_channel_labels else {}
            channel_meta.append(
                {
                    "channel_name_norm": name,
                    "raw_channel_name": raw_channel_names[source_idx] if source_idx < len(raw_channel_names) else name,
                    "contact_group": group,
                    "contact_number": number,
                    "dataset": "pediatric",
                    "source_center": "pediatric",
                    "source_path": str(run.edf_path),
                    "csv_path": str(run.csv_path),
                    "label_source": str(xlsx_label.get("label_source", "pediatric_channel_level_xlsx")) if subject_channel_labels else "pediatric_xlsx_candidate_verified_against_edf",
                    "label_mode": label_mode,
                    "label_description_raw": str(xlsx_label.get("source_description", label_desc)) if subject_channel_labels else str(label_desc),
                    "bad_channel_description_raw": str(bad_desc),
                    "is_ez_or_soz": int(name in matched_ez),
                    "usable_channel_mask": int(xlsx_label.get("usable", 1)) if subject_channel_labels else 1,
                    "is_monitoring_bad": int(xlsx_label.get("is_monitoring_bad", 0)) if subject_channel_labels else int(name in matched_bad),
                    "final_label": float(labels[out_idx]),
                    "unmatched_ez_channels": unmatched_ez,
                    "uncertain_ez_tokens": uncertain_ez,
                    "uncertain_bad_tokens": uncertain_bad,
                    "unlabeled_edf_channel_count": len(unlabeled_edf_channels),
                    "xlsx_label_channels_not_in_edf_count": len(label_channels_not_in_edf),
                    "surgery_success": success,
                    "metadata_sheet": run.metadata.get("metadata_sheet"),
                }
            )
        onset_sec = (run.onset_timestamp - run.recording_start_timestamp).total_seconds()
        offset_sec = (run.offset_timestamp - run.recording_start_timestamp).total_seconds()
        duration_sec = float(kept_data.shape[1]) / max(float(sfreq), 1e-8)
        if onset_sec <= 0.0 or onset_sec >= duration_sec:
            continue
        offset_sec = min(max(offset_sec, onset_sec + 1.0), duration_sec)
        seizures.append(
            PediatricSeizureRecord(
                subject_id=run.subject_id,
                seizure_id=f"{run.subject_id}__{run.edf_path.stem}",
                signal=kept_data,
                sfreq=sfreq,
                channel_names=kept_channel_names,
                seizure_onset_sec=float(onset_sec),
                seizure_offset_sec=float(offset_sec),
                labels=labels,
                channel_meta=channel_meta,
            )
        )
    return _build_pediatric_patient_records(seizures)


__all__ = [
    "PediatricEvent",
    "PediatricPatientRecord",
    "PediatricRun",
    "PediatricSeizureRecord",
    "discover_pediatric_runs",
    "expand_channel_description",
    "load_pediatric_channel_labels",
    "load_pediatric_patient_records",
    "load_subject_metadata",
    "read_event_csv",
    "recording_start_from_csv",
    "seizure_interval_from_csv",
]
