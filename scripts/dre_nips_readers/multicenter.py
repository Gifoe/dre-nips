from __future__ import annotations

from .audit import ReadAudit
from .bids_loader import load_bids_patient_records
from .schemas import DataInterfaceConfig, PatientRecord


def load_multicenter_patient_records(cfg: DataInterfaceConfig, audit: ReadAudit | None = None) -> list[PatientRecord]:
    return load_bids_patient_records(
        root=cfg.multicenter_root,
        participants_path=cfg.multicenter_participants_path,
        sidecar_root=cfg.multicenter_sidecar_root,
        dataset_name="multicenter",
        cfg=cfg,
        audit=audit,
    )
