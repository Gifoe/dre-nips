from pathlib import Path

import numpy as np
import pytest

from biodynformer.feature_bank import build_feature_bank_from_records, load_feature_bank_index


@pytest.fixture()
def small_feature_bank(tmp_path: Path):
    records = []
    for idx, center in enumerate(["lzu", "hup", "multicenter", "pediatric"], start=1):
        for outcome in [True, False]:
            samples = 1300
            t = np.linspace(0, 20 + idx, samples)
            signal = np.vstack([np.sin(t), np.cos(t), np.sin(t * 2.0)]).astype(np.float32)
            records.append(
                {
                    "center": center,
                    "subject_id": f"S{idx}_{int(outcome)}",
                    "outcome_success": outcome,
                    "seizures": [
                        {
                            "run_id": "run1",
                            "seizure_id": "sz1",
                            "quality_rating": "GOOD",
                            "signal": signal,
                            "sfreq": 10.0,
                            "seizure_onset_sec": 125.0,
                            "channel_names": ["A1", "A2", "B1"],
                            "labels_ez": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                        }
                    ],
                }
            )
    build_feature_bank_from_records(records, output_dir=tmp_path)
    return load_feature_bank_index(tmp_path)
