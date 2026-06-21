import subprocess
import sys
from pathlib import Path


def test_build_feature_bank_cli_help_imports_package():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "build_feature_bank.py"), "--help"],
        cwd=root,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Build one reusable preictal" in result.stdout
