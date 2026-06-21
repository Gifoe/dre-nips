from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .features import (
    COVERAGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    QUALITY_FEATURE_NAMES,
    compute_causal_edge,
    compute_coverage_features,
    compute_node_features,
    compute_quality_features,
    compute_structural_edge,
    compute_sync_edge,
)
from .hfo import HFO_FEATURE_NAMES, compute_hfo_features
from .preictal_windows import DEFAULT_WINDOWS, extract_preictal_segments
from .quality_filter import filter_patient_records


CENTERS = ("lzu", "hup", "multicenter", "pediatric")
FORBIDDEN_BANK_KEYS = {
    "has_" + "inter" + "ictal",
    "inter" + "ictal_missing_mask",
    "ictal_vs_" + "inter" + "ictal",
}
FEATURE_SCHEMA = {
    "windows": [{"name": w.name, "start_sec": w.start_sec, "end_sec": w.end_sec} for w in DEFAULT_WINDOWS],
    "node_features": NODE_FEATURE_NAMES,
    "hfo_features": HFO_FEATURE_NAMES,
    "quality_features": QUALITY_FEATURE_NAMES,
    "coverage_features": COVERAGE_FEATURE_NAMES,
}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "success", "successful", "s"}:
        return True
    if text in {"0", "false", "no", "n", "failure", "failed", "f"}:
        return False
    if text in {"i", "engel i", "engeli", "engel1", "engel 1"}:
        return True
    if text in {"ii", "iii", "iv", "engel ii", "engel iii", "engel iv", "engel2", "engel3", "engel4"}:
        return False
    return None


def _iter_meta_dicts(obj: Any) -> list[Mapping[str, Any]]:
    metas = []
    if isinstance(obj, Mapping):
        if isinstance(obj.get("channel_meta"), list):
            metas.extend(item for item in obj["channel_meta"] if isinstance(item, Mapping))
        if isinstance(obj.get("metadata"), Mapping):
            metas.append(obj["metadata"])
    else:
        channel_meta = getattr(obj, "channel_meta", None)
        if isinstance(channel_meta, list):
            metas.extend(item for item in channel_meta if isinstance(item, Mapping))
        metadata = getattr(obj, "metadata", None)
        if isinstance(metadata, Mapping):
            metas.append(metadata)
    return metas


def extract_patient_outcome(patient: Any) -> bool | None:
    for key in ("outcome_success", "surgery_success", "success_used", "is_successful", "outcome"):
        value = _get(patient, key, None)
        parsed = _as_bool_or_none(value)
        if parsed is not None:
            return parsed
    for meta in _iter_meta_dicts(patient):
        for key in (
            "success_used",
            "surgery_success",
            "is_successful",
            "outcome_success",
            "outcome",
            "lzu_outcome_raw",
            "手术结果（成功/失败）",
            "手术结果",
        ):
            parsed = _as_bool_or_none(meta.get(key))
            if parsed is not None:
                return parsed
    return None


def _subject_id(center: str, subject_id: Any) -> str:
    text = str(subject_id).strip()
    if text.startswith(f"{center}:"):
        return text
    return f"{center}:{text}"


def _safe_run_id(value: Any) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip() or "run")


def _feature_arrays_for_seizure(seizure: Any) -> dict[str, np.ndarray]:
    signal = np.asarray(_get(seizure, "signal"), dtype=np.float32)
    sfreq = float(_get(seizure, "sfreq"))
    onset_sec = float(_get(seizure, "seizure_onset_sec", _get(seizure, "onset_sec")))
    channel_names = [str(name) for name in _get(seizure, "channel_names")]
    segments, window_mask = extract_preictal_segments(signal, sfreq=sfreq, onset_sec=onset_sec)
    valid_count = int(window_mask.sum())
    channels = signal.shape[0]
    node = np.zeros((len(DEFAULT_WINDOWS), channels, len(NODE_FEATURE_NAMES)), dtype=np.float32)
    hfo = np.zeros((len(DEFAULT_WINDOWS), channels, len(HFO_FEATURE_NAMES)), dtype=np.float32)
    quality = np.zeros((len(DEFAULT_WINDOWS), channels, len(QUALITY_FEATURE_NAMES)), dtype=np.float32)
    causal = np.zeros((len(DEFAULT_WINDOWS), channels, channels), dtype=np.float32)
    sync = np.zeros((len(DEFAULT_WINDOWS), channels, channels), dtype=np.float32)
    for idx, segment in enumerate(segments):
        if not window_mask[idx]:
            continue
        node[idx] = compute_node_features(segment, sfreq)
        hfo[idx] = compute_hfo_features(segment, sfreq)
        quality[idx] = compute_quality_features(segment, sfreq, valid_count)
        causal[idx] = compute_causal_edge(segment)
        sync[idx] = compute_sync_edge(segment)
    return {
        "node_features": node,
        "hfo_features": hfo,
        "quality_features": quality,
        "causal_edge": causal,
        "sync_edge": sync,
        "structural_edge": compute_structural_edge(channel_names),
        "coverage_features": compute_coverage_features(channel_names),
        "window_mask": window_mask,
        "channel_mask": np.ones((channels,), dtype=bool),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_feature_bank_from_records(
    patients: Sequence[Any],
    *,
    output_dir: str | Path,
    quality_filter: bool = True,
    keep_ratings: str | Sequence[str] | None = None,
    drop_ratings: str | Sequence[str] | None = None,
    missing_policy: str = "drop",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    patients_in = list(patients)
    if quality_filter:
        patients_kept, diagnostics = filter_patient_records(
            patients_in,
            keep_ratings=keep_ratings,
            drop_ratings=drop_ratings,
            missing_policy=missing_policy,
        )
    else:
        patients_kept = patients_in
        diagnostics = []
    _write_csv(output / "quality_filter_diagnostics.csv", diagnostics)

    patient_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for patient in patients_kept:
        center = str(_get(patient, "center", _get(patient, "source_center", ""))).strip().lower()
        if center not in CENTERS:
            raise ValueError(f"Unsupported center {center!r}; expected {CENTERS}.")
        subject = _subject_id(center, _get(patient, "subject_id", ""))
        outcome = extract_patient_outcome(patient)
        seizures = list(_get(patient, "seizures", []) or [])
        patient_rows.append(
            {
                "center": center,
                "subject_id": subject,
                "outcome_success": "" if outcome is None else int(outcome),
                "num_retained_runs": len(seizures),
            }
        )
        for seizure in seizures:
            run_id = _safe_run_id(_get(seizure, "run_id", _get(seizure, "seizure_id", "run")))
            channel_names = np.asarray([str(name) for name in _get(seizure, "channel_names")], dtype=object)
            labels = np.asarray(_get(seizure, "labels_ez", _get(seizure, "labels")), dtype=np.float32)
            arrays = _feature_arrays_for_seizure(seizure)
            tensor_dir = output / "tensors" / center / subject.replace(":", "_")
            tensor_dir.mkdir(parents=True, exist_ok=True)
            tensor_path = tensor_dir / f"{run_id}.npz"
            np.savez_compressed(
                tensor_path,
                **arrays,
                labels_ez=labels,
                outcome_success=np.asarray(-1 if outcome is None else int(outcome), dtype=np.int8),
                center=np.asarray(center),
                subject_id=np.asarray(subject),
                run_id=np.asarray(run_id),
                channel_names=channel_names,
            )
            run_rows.append(
                {
                    "center": center,
                    "subject_id": subject,
                    "run_id": run_id,
                    "tensor_path": str(tensor_path),
                    "outcome_success": "" if outcome is None else int(outcome),
                    "num_channels": int(labels.shape[0]),
                    "num_valid_windows": int(arrays["window_mask"].sum()),
                }
            )

    (output / "feature_schema.json").write_text(json.dumps(FEATURE_SCHEMA, indent=2), encoding="utf-8")
    _write_csv(output / "patient_manifest.csv", patient_rows)
    _write_csv(output / "run_manifest.csv", run_rows)
    summary = {"num_patients": len(patient_rows), "num_runs": len(run_rows), "output_dir": str(output)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_feature_bank_index(feature_bank_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(feature_bank_dir) / "run_manifest.csv"
    with open(path, "r", encoding="utf-8-sig", newline="") as fin:
        rows = list(csv.DictReader(fin))
    for row in rows:
        row["tensor_path"] = str(Path(row["tensor_path"]))
        row["outcome_success"] = None if row.get("outcome_success", "") == "" else bool(int(row["outcome_success"]))
        row["num_channels"] = int(row.get("num_channels", 0) or 0)
        row["num_valid_windows"] = int(row.get("num_valid_windows", 0) or 0)
    return rows


def load_records_json(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("records JSON must contain a list of patient records.")
    for patient in raw:
        for seizure in patient.get("seizures", []):
            if isinstance(seizure.get("signal"), list):
                seizure["signal"] = np.asarray(seizure["signal"], dtype=np.float32)
            if isinstance(seizure.get("labels_ez"), list):
                seizure["labels_ez"] = np.asarray(seizure["labels_ez"], dtype=np.float32)
    return raw


def load_patient_records_pickle(path: str | Path) -> list[Any]:
    with open(path, "rb") as fin:
        payload = pickle.load(fin)
    if not isinstance(payload, list):
        raise TypeError("patient records pickle must contain a list.")
    return payload

