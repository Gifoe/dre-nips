from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def inspect_patient_records(records: list[Any]) -> dict[str, Any]:
    centers: dict[str, int] = {}
    seizures = 0
    with_outcome = 0
    label_values: set[float] = set()
    errors: list[str] = []
    for patient in records:
        center = str(_get(patient, "center", "unknown"))
        centers[center] = centers.get(center, 0) + 1
        if _get(patient, "outcome_success", _get(patient, "Engel", None)) is not None:
            with_outcome += 1
        patient_seizures = list(_get(patient, "seizures", []) or [])
        seizures += len(patient_seizures)
        labels = np.asarray(_get(patient, "labels", _get(patient, "labels_nez", [])), dtype=np.float32)
        label_values.update(float(x) for x in np.unique(labels) if np.isfinite(x))
        if not patient_seizures:
            errors.append(f"{_get(patient, 'subject_id', '?')}: no seizures")
        for seizure in patient_seizures:
            signal = np.asarray(_get(seizure, "signal", []))
            if signal.ndim != 2:
                errors.append(f"{_get(patient, 'subject_id', '?')}/{_get(seizure, 'run_id', '?')}: signal is not [C,N]")
            if float(_get(seizure, "sfreq", 0.0) or 0.0) <= 0.0:
                errors.append(f"{_get(patient, 'subject_id', '?')}/{_get(seizure, 'run_id', '?')}: invalid sfreq")
    return {
        "patients": len(records),
        "seizures": seizures,
        "centers": centers,
        "patients_with_outcome": with_outcome,
        "label_values": sorted(label_values),
        "label_semantics_expected": "1=NEZ, 0=EZ, -1=unknown",
        "errors": errors,
        "usable_patient_records": len(records) > 0 and seizures > 0 and not errors,
    }


def load_patient_records(path: Path) -> list[Any]:
    with open(path, "rb") as fin:
        payload = pickle.load(fin)
    if isinstance(payload, Mapping):
        payload = payload.get("records", payload.get("patient_records", payload))
    if not isinstance(payload, list):
        raise TypeError("patient_records.pkl must contain a list or a mapping with records.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PGC-SEEG patient_records.pkl.")
    parser.add_argument("--patient-records-pkl", "--patient_records_pkl", dest="patient_records_pkl", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_patient_records(load_patient_records(args.patient_records_pkl))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
