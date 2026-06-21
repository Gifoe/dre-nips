from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .schemas import PatientRecord, SeizureRecord, validate_seizure_shape


class ReadAudit:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, center_id: str, row_type: str, **row: Any) -> None:
        self.rows.append({"center_id": center_id, "row_type": row_type, **row})

    def add_skipped_patient(self, center_id: str, subject_id_value: str, reason: str, **row: Any) -> None:
        for key in ("subject_id", "skipped", "skip_reason"):
            row.pop(key, None)
        self.add(center_id, "patient", subject_id=subject_id_value, skipped=True, skip_reason=reason, **row)

    def add_skipped_seizure(
        self,
        center_id: str,
        subject_id_value: str,
        seizure_id_value: str,
        reason: str,
        **row: Any,
    ) -> None:
        for key in ("subject_id", "seizure_id", "skipped", "skip_reason"):
            row.pop(key, None)
        self.add(
            center_id,
            "seizure",
            subject_id=subject_id_value,
            seizure_id=seizure_id_value,
            skipped=True,
            skip_reason=reason,
            **row,
        )

    def add_channel(self, center_id: str, subject_id: str, seizure_id: str | None, **row: Any) -> None:
        self.add(center_id, "channel", subject_id=subject_id, seizure_id=seizure_id, **row)

    def add_loaded_records(self, center_id: str, patients: Sequence[PatientRecord]) -> None:
        for patient in patients:
            self.add(center_id, "patient", **_patient_audit_row(center_id, patient))
            for seizure in patient.seizures:
                self.add(center_id, "seizure", **_seizure_audit_row(center_id, seizure))

    def validate_and_filter(self, center_id: str, patients: Sequence[PatientRecord], strict: bool = False) -> list[PatientRecord]:
        kept: list[PatientRecord] = []
        for patient in patients:
            reason = _patient_failure_reason(patient)
            if reason is not None:
                row = _patient_audit_row(center_id, patient)
                for key in ("subject_id", "skipped", "skip_reason"):
                    row.pop(key, None)
                self.add_skipped_patient(center_id, patient.subject_id, reason, **row)
                if strict:
                    raise ValueError(f"{center_id}/{patient.subject_id}: {reason}")
                continue
            kept.append(patient)
        return kept

    def write(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.rows)
        if df.empty:
            df = pd.DataFrame(columns=["center_id", "row_type"])

        centers = ["lzu", "hup", "multicenter"]
        for center in centers:
            center_df = df[df["center_id"].eq(center)].copy()
            center_df.to_csv(output_dir / f"{center}_read_audit.csv", index=False)
        df.to_csv(output_dir / "all_centers_read_audit.csv", index=False)

        summary = {
            "output_dir": str(output_dir),
            "n_rows": int(len(df)),
            "by_center": {},
            "by_row_type": {str(k): int(v) for k, v in Counter(df.get("row_type", [])).items()},
        }
        for center in centers:
            center_df = df[df["center_id"].eq(center)]
            patient_df = center_df[center_df["row_type"].eq("patient")]
            seizure_df = center_df[center_df["row_type"].eq("seizure")]
            summary["by_center"][center] = {
                "audit_rows": int(len(center_df)),
                "loaded_patients": int(patient_df["skipped"].fillna(False).eq(False).sum())
                if "skipped" in patient_df
                else int(len(patient_df)),
                "skipped_patients": int(patient_df["skipped"].fillna(False).eq(True).sum())
                if "skipped" in patient_df
                else 0,
                "loaded_seizures": int(seizure_df["skipped"].fillna(False).eq(False).sum())
                if "skipped" in seizure_df
                else int(len(seizure_df)),
                "skipped_seizures": int(seizure_df["skipped"].fillna(False).eq(True).sum())
                if "skipped" in seizure_df
                else 0,
            }
        with open(output_dir / "read_audit_summary.json", "w", encoding="utf-8") as fout:
            json.dump(summary, fout, indent=2, ensure_ascii=False)


def _patient_audit_row(center_id: str, patient: PatientRecord) -> dict[str, Any]:
    labels = np.asarray(patient.labels, dtype=np.float32)
    first_meta = patient.channel_meta[0] if patient.channel_meta else {}
    source_paths = sorted(
        {
            str(meta.get("source_path"))
            for seizure in patient.seizures
            for meta in seizure.channel_meta[:1]
            if meta.get("source_path")
        }
    )
    run_ids = sorted(
        {
            str(meta.get("run_id", seizure.seizure_id))
            for seizure in patient.seizures
            for meta in seizure.channel_meta[:1]
        }
    )
    bad_overlap = _unique_meta_values(patient.channel_meta, "bad_ez_overlap_ids")
    unmatched_label_ids = _unique_meta_values(patient.channel_meta, "unmatched_label_ids")
    unmatched_channels = _unique_meta_values(patient.channel_meta, "unmatched_channels")
    n_ez = int((labels == 0.0).sum())
    n_nez = int((labels == 1.0).sum())
    return {
        "subject_id": patient.subject_id,
        "global_subject_id": f"{center_id}:{patient.subject_id}",
        "n_edf": len(source_paths),
        "n_runs": len(run_ids),
        "participant_id": first_meta.get("participant_id", patient.subject_id),
        "outcome": first_meta.get("outcome"),
        "success_used": first_meta.get("success_used"),
        "n_ictal_runs": _first_int_meta(patient.channel_meta, "n_ictal_runs_subject") or len(run_ids),
        "n_loaded_seizures": len(patient.seizures),
        "n_seizures": len(patient.seizures),
        "n_channels_total": len(patient.canonical_channels),
        "n_valid_channels": len(patient.canonical_channels),
        "n_bad_channels": _first_int_meta(patient.channel_meta, "n_bad_channels_source"),
        "n_ez_channels_label0": n_ez,
        "n_nez_channels_label1": n_nez,
        "ez_ratio": float(n_ez / labels.size) if labels.size else 0.0,
        "n_unmatched_channels": len(unmatched_channels),
        "unmatched_channels": ";".join(unmatched_channels),
        "n_unmatched_label_ids": len(unmatched_label_ids),
        "unmatched_label_ids": ";".join(unmatched_label_ids),
        "bad_ez_overlap": bool(bad_overlap),
        "bad_ez_overlap_ids": ";".join(bad_overlap),
        "skipped": False,
        "skip_reason": None,
    }


def _seizure_audit_row(center_id: str, seizure: SeizureRecord) -> dict[str, Any]:
    first_meta = seizure.channel_meta[0] if seizure.channel_meta else {}
    raw_duration = _safe_float(first_meta.get("raw_duration_sec"))
    onset = float(seizure.seizure_onset_sec) if seizure.seizure_onset_sec is not None else None
    offset = float(seizure.seizure_offset_sec) if seizure.seizure_offset_sec is not None else None
    onset_valid = onset is not None and (raw_duration is None or 0.0 <= onset <= raw_duration)
    offset_valid = offset is None or (raw_duration is None or 0.0 <= offset <= raw_duration)
    return {
        "subject_id": seizure.subject_id,
        "seizure_id": seizure.seizure_id,
        "edf_path": first_meta.get("source_path"),
        "channels_path": first_meta.get("channels_path"),
        "events_path": first_meta.get("events_path"),
        "onset_sec": onset,
        "offset_sec": offset,
        "raw_duration_sec": raw_duration,
        "onset_valid": onset_valid,
        "offset_valid": offset_valid,
        "n_signal_channels": int(seizure.signal.shape[0]),
        "n_label_channels": int(seizure.labels.shape[0]),
        "sfreq_original": first_meta.get("sfreq_original"),
        "sfreq_after_resample": float(seizure.sfreq),
        "event_index": first_meta.get("event_index"),
        "event_trial_type_used": first_meta.get("event_trial_type_used"),
        "skipped": False,
        "skip_reason": None,
    }


def _patient_failure_reason(patient: PatientRecord) -> str | None:
    if not patient.seizures:
        return "no_seizures"
    labels = np.asarray(patient.labels, dtype=np.float32)
    if not (labels == 0.0).any():
        return "no_ez_label0_channels"
    if not (labels == 1.0).any():
        return "no_nez_label1_channels"
    for seizure in patient.seizures:
        try:
            validate_seizure_shape(seizure)
        except ValueError as exc:
            return str(exc)
        first_meta = seizure.channel_meta[0] if seizure.channel_meta else {}
        raw_duration = _safe_float(first_meta.get("raw_duration_sec"))
        onset = seizure.seizure_onset_sec
        if onset is None:
            return f"{seizure.seizure_id}: onset_missing"
        if raw_duration is not None and not (0.0 <= float(onset) <= raw_duration):
            return f"{seizure.seizure_id}: onset_out_of_range"
    return None


def _unique_meta_values(metas: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    values: set[str] = set()
    for meta in metas:
        value = meta.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, (list, tuple, set)):
            values.update(str(item) for item in value)
        else:
            values.add(str(value))
    return sorted(values)


def _first_int_meta(metas: Sequence[Mapping[str, Any]], key: str) -> int:
    for meta in metas:
        value = meta.get(key)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result
