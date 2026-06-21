from pathlib import Path

from scripts.dre_nips_readers.bids_loader import load_bids_patient_records
from scripts.dre_nips_readers.schemas import DataInterfaceConfig


def test_bids_reader_uses_participants_scope_when_success_only_is_false(monkeypatch, tmp_path: Path):
    participants_path = tmp_path / "participants.tsv"
    participants_path.write_text(
        "participant_id\toutcome\n"
        "sub-HUP001\tS\n"
        "sub-HUP002\tF\n",
        encoding="utf-8",
    )
    captured = {}

    def fake_discover(*, root, participants, subject_filter):
        captured["participants"] = participants
        return [], {}

    monkeypatch.setattr("scripts.dre_nips_readers.bids_loader.discover_bids_edfs_for_participants", fake_discover)

    records = load_bids_patient_records(
        root=tmp_path,
        participants_path=participants_path,
        sidecar_root=None,
        dataset_name="hup",
        cfg=DataInterfaceConfig(success_only=False, write_read_audit=False),
    )

    assert records == []
    assert captured["participants"] == ["sub-HUP001", "sub-HUP002"]
