from biodynformer.quality_filter import filter_patient_records


def test_quality_filter_drops_poor_seizure_but_keeps_patient_with_good_seizure():
    patients = [
        {
            "center": "lzu",
            "subject_id": "S1",
            "seizures": [
                {"run_id": "r1", "seizure_id": "sz1", "quality_rating": "GOOD"},
                {"run_id": "r2", "seizure_id": "sz2", "quality_rating": "POOR"},
            ],
        }
    ]

    kept, diagnostics = filter_patient_records(patients)

    assert len(kept) == 1
    assert [s["run_id"] for s in kept[0]["seizures"]] == ["r1"]
    assert {row["action"] for row in diagnostics} == {"keep", "drop"}


def test_quality_filter_removes_patient_when_all_seizures_are_dropped():
    patients = [
        {
            "center": "hup",
            "subject_id": "S2",
            "seizures": [{"run_id": "r1", "seizure_id": "sz1", "quality_rating": "POOR"}],
        }
    ]

    kept, diagnostics = filter_patient_records(patients)

    assert kept == []
    assert diagnostics[0]["reason"] == "quality_rating_dropped"


def test_quality_filter_missing_match_defaults_to_drop():
    patients = [
        {
            "center": "pediatric",
            "subject_id": "S3",
            "seizures": [{"run_id": "r1", "seizure_id": "sz1"}],
        }
    ]

    kept, diagnostics = filter_patient_records(patients)

    assert kept == []
    assert diagnostics[0]["reason"] == "missing_quality_match"
