from __future__ import annotations

import copy
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_KEEP_RATINGS = {"GOOD", "REVIEW"}
DEFAULT_DROP_RATINGS = {"POOR"}


@dataclass(frozen=True)
class QualityDecision:
    center: str
    subject_id: str
    seizure_id: str
    run_id: str
    action: str
    reason: str
    quality_rating: str
    quality_match_key: str
    report_path: str = ""
    report_row: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "center": self.center,
            "subject_id": self.subject_id,
            "seizure_id": self.seizure_id,
            "run_id": self.run_id,
            "action": self.action,
            "reason": self.reason,
            "quality_rating": self.quality_rating,
            "quality_match_key": self.quality_match_key,
            "report_path": self.report_path,
            "report_row": self.report_row,
        }


def parse_rating_set(value: str | Sequence[str] | None, default: set[str]) -> set[str]:
    if value is None:
        return set(default)
    if isinstance(value, str):
        parts = re.split(r"[,;\s]+", value)
    else:
        parts = [str(part) for part in value]
    parsed = {part.strip().upper() for part in parts if part and part.strip()}
    return parsed or set(default)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(value).lower())


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def _copy_patient_with_seizures(patient: Any, seizures: list[Any]) -> Any:
    if isinstance(patient, dict):
        cloned = dict(patient)
        cloned["seizures"] = seizures
        return cloned
    cloned = copy.copy(patient)
    setattr(cloned, "seizures", seizures)
    return cloned


def seizure_quality_keys(center: str, subject_id: Any, seizure: Any) -> list[str]:
    subject_key = normalize_key(subject_id)
    run_id = normalize_key(_get(seizure, "run_id", ""))
    seizure_id = normalize_key(_get(seizure, "seizure_id", ""))
    keys: list[str] = []
    if run_id and seizure_id:
        keys.append("|".join([center, subject_key, run_id, seizure_id]))
    if run_id:
        keys.append("|".join([center, subject_key, run_id]))
    if seizure_id:
        keys.append("|".join([center, subject_key, seizure_id]))
    keys.append("|".join([center, subject_key]))
    return keys


def build_quality_index(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        center = normalize_text(row.get("center", "")).lower()
        subject_id = row.get("subject_id", row.get("patient_id", ""))
        run_id = row.get("run_id", "")
        seizure_id = row.get("seizure_id", row.get("seizure_name", ""))
        probe = {"run_id": run_id, "seizure_id": seizure_id}
        for key in seizure_quality_keys(center, subject_id, probe):
            index.setdefault(key, row)
    return index


def read_quality_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as fin:
        return list(csv.DictReader(fin))


def filter_patient_records(
    patients: Sequence[Any],
    *,
    quality_index: Mapping[str, Mapping[str, Any]] | None = None,
    keep_ratings: str | Sequence[str] | None = None,
    drop_ratings: str | Sequence[str] | None = None,
    missing_policy: str = "drop",
) -> tuple[list[Any], list[dict[str, str]]]:
    keep = parse_rating_set(keep_ratings, DEFAULT_KEEP_RATINGS)
    drop = parse_rating_set(drop_ratings, DEFAULT_DROP_RATINGS)
    missing_policy = str(missing_policy).strip().lower()
    if missing_policy not in {"drop", "keep"}:
        raise ValueError("missing_policy must be 'drop' or 'keep'.")

    kept_patients: list[Any] = []
    diagnostics: list[dict[str, str]] = []
    qindex = quality_index or {}

    for patient in patients:
        center = normalize_text(_get(patient, "center", _get(patient, "source_center", ""))).lower()
        subject_id = normalize_text(_get(patient, "subject_id", ""))
        kept_seizures: list[Any] = []
        for seizure in list(_get(patient, "seizures", []) or []):
            run_id = normalize_text(_get(seizure, "run_id", ""))
            seizure_id = normalize_text(_get(seizure, "seizure_id", ""))
            rating = normalize_text(_get(seizure, "quality_rating", "")).upper()
            match_key = ""
            row = None
            if not rating:
                for key in seizure_quality_keys(center, subject_id, seizure):
                    row = qindex.get(key)
                    if row is not None:
                        match_key = key
                        rating = normalize_text(row.get("quality_rating", row.get("rating", ""))).upper()
                        break
            if not rating:
                action = "keep" if missing_policy == "keep" else "drop"
                reason = "missing_quality_match"
            elif rating in drop:
                action = "drop"
                reason = "quality_rating_dropped"
            elif rating in keep:
                action = "keep"
                reason = "quality_rating_allowed"
            else:
                action = "drop"
                reason = "quality_rating_not_allowed"

            diagnostics.append(
                QualityDecision(
                    center=center,
                    subject_id=subject_id,
                    seizure_id=seizure_id,
                    run_id=run_id,
                    action=action,
                    reason=reason,
                    quality_rating=rating,
                    quality_match_key=match_key,
                    report_path=normalize_text(row.get("report_path", "")) if row is not None else "",
                    report_row=normalize_text(row.get("report_row", "")) if row is not None else "",
                ).as_dict()
            )
            if action == "keep":
                _set(seizure, "quality_rating", rating or "UNRATED")
                kept_seizures.append(seizure)
        if kept_seizures:
            kept_patients.append(_copy_patient_with_seizures(patient, kept_seizures))
    return kept_patients, diagnostics
