from __future__ import annotations

from pathlib import Path

from .common import PatientRecord, load_manifest_records


def load_pediatric_records(*, manifest_path: str | Path | None = None, root: str | Path | None = None) -> list[PatientRecord]:
    return load_manifest_records("pediatric", manifest_path=manifest_path, root=root)
