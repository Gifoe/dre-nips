from __future__ import annotations

from .audit import ReadAudit
from .bids_loader import load_bids_patient_records
from .schemas import DataInterfaceConfig, PatientRecord


def load_hup_patient_records(cfg: DataInterfaceConfig, audit: ReadAudit | None = None) -> list[PatientRecord]:
    return load_bids_patient_records(
        root=cfg.hup_root,
        participants_path=cfg.hup_participants_path,
        sidecar_root=None,
        dataset_name="hup",
        cfg=cfg,
        audit=audit,
    )
