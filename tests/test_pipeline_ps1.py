from pathlib import Path


def test_dre_nips_pipeline_script_contains_ordered_steps_and_defaults():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_dre_nips_preictal_pipeline.ps1"
    text = script.read_text(encoding="utf-8")

    reader = text.index("build_patient_records_from_dre_nips_preictal_only.py")
    feature_bank = text.index("build_feature_bank.py")
    run_all = text.index("run_all_versions.py")

    assert reader < feature_bank < run_all
    assert "0x5170,0x5927,0x4E8C,0x9662,0x65B0" in text
    assert "E:\\DRE-nips\\dataest" in text
    assert "0x513F,0x79D1,0x533B,0x9662" in text
    assert "D:\\all_seeg_data" in text
    assert "--no-success-only" in text
    assert "--source" in text
    assert "patient-records-pkl" in text
    assert "DreNipsRoot" not in text
