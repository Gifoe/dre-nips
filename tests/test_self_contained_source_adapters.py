import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from biodynformer.source_adapters import load_four_center_records
from scripts.build_feature_bank import build_parser


def _write_center_manifest(root: Path, center: str, outcome: str = "S") -> Path:
    center_dir = root / center
    center_dir.mkdir(parents=True)
    signal_path = center_dir / "signal.npy"
    np.save(signal_path, np.ones((2, 1300), dtype=np.float32))
    manifest_path = center_dir / "manifest.csv"
    with open(manifest_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=[
                "subject_id",
                "run_id",
                "signal_path",
                "sfreq",
                "seizure_onset_sec",
                "channel_names",
                "labels_ez",
                "outcome",
                "quality_rating",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "subject_id": f"{center}_001",
                "run_id": "run1",
                "signal_path": str(signal_path),
                "sfreq": "10",
                "seizure_onset_sec": "125",
                "channel_names": "A1,A2",
                "labels_ez": "1,0",
                "outcome": outcome,
                "quality_rating": "GOOD",
            }
        )
    return manifest_path


def test_build_parser_no_longer_exposes_dre_nips_runtime_dependency():
    parser = build_parser()
    help_text = parser.format_help()

    assert "dre-nips-root" not in help_text
    assert "source_adapters" in help_text
    assert parser.parse_args(["--output-dir", "D:\\tmp\\bank"]).source == "four-center-raw"


def test_four_center_adapter_loads_manifest_records_without_github(tmp_path: Path):
    manifest = _write_center_manifest(tmp_path, "lzu", outcome="F")

    records = load_four_center_records(centers=["lzu"], manifest_paths={"lzu": manifest})

    assert len(records) == 1
    assert records[0].center == "lzu"
    assert records[0].outcome_success is False
    assert records[0].seizures[0].signal.shape == (2, 1300)
    assert records[0].seizures[0].labels_ez.tolist() == [1.0, 0.0]


def test_cli_help_does_not_mention_github_reader():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "build_feature_bank.py"), "--help"],
        cwd=root,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "dre-nips-root" not in result.stdout
