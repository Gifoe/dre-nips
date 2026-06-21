from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import PatientRecord, SeizureRecord
from .hup_reader import load_hup_records
from .lzu_reader import load_lzu_records
from .multicenter_reader import load_multicenter_records
from .pediatric_reader import load_pediatric_records


LOADERS = {
    "lzu": load_lzu_records,
    "hup": load_hup_records,
    "multicenter": load_multicenter_records,
    "pediatric": load_pediatric_records,
}


def load_four_center_records(
    *,
    centers: Sequence[str],
    manifest_paths: Mapping[str, str | Path | None] | None = None,
    roots: Mapping[str, str | Path | None] | None = None,
) -> list[PatientRecord]:
    manifests = dict(manifest_paths or {})
    root_map = dict(roots or {})
    records: list[PatientRecord] = []
    for center in centers:
        center_key = str(center).strip().lower()
        loader = LOADERS.get(center_key)
        if loader is None:
            raise ValueError(f"Unknown center {center!r}; expected {sorted(LOADERS)}.")
        records.extend(loader(manifest_path=manifests.get(center_key), root=root_map.get(center_key)))
    return records


__all__ = ["PatientRecord", "SeizureRecord", "load_four_center_records"]
