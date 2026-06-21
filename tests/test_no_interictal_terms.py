from pathlib import Path


def test_source_and_configs_do_not_use_removed_reference_terms():
    root = Path(__file__).resolve().parents[1]
    forbidden = [
        "inter" + "ictal",
        "has_" + "inter" + "ictal",
        "ictal_vs_" + "inter" + "ictal",
    ]
    checked = []
    for folder in [root / "biodynformer", root / "configs"]:
        for path in folder.rglob("*"):
            if path.suffix.lower() not in {".py", ".yaml", ".yml", ".json"}:
                continue
            checked.append(path)
            text = path.read_text(encoding="utf-8").lower()
            for term in forbidden:
                assert term not in text, f"{term!r} found in {path}"
    assert checked
