import subprocess
import sys
from pathlib import Path

from biodynformer.pipeline import ManifestInputError, resolve_required_manifests


def test_resolve_required_manifests_reports_missing_root_manifest(tmp_path: Path):
    root = tmp_path / "lzu"
    root.mkdir()

    try:
        resolve_required_manifests(["lzu"], manifest_paths={}, roots={"lzu": root})
    except ManifestInputError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected missing manifest error")

    assert str(root / "manifest.csv") in message
    assert "subject_id, run_id, signal_path" in message


def test_resolve_required_manifests_accepts_explicit_manifest(tmp_path: Path):
    manifest = tmp_path / "lzu_manifest.csv"
    manifest.write_text("subject_id,run_id,signal_path\n", encoding="utf-8")

    resolved = resolve_required_manifests(["lzu"], manifest_paths={"lzu": manifest}, roots={})

    assert resolved == {"lzu": manifest}


def test_run_full_pipeline_cli_help_imports_package():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "run_full_pipeline.py"), "--help"],
        cwd=root,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "audit, build, audit, and run" in result.stdout
