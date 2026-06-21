# PGC-SEEG Multitask Pipeline

This folder rebuilds the `dre-aaai`/`dre-nips` code line as a self-contained
PGC-SEEG pipeline while preserving the copied B0-Pruned-EZBackbone baseline.

## Label Semantics

Task1 keeps the existing baseline training convention:

```text
1 = NEZ
0 = EZ
-1 = unknown / masked
```

The model trains on `labels_nez` and internal `P(NEZ)`. Reports and prediction
CSVs expose EZ localization as:

```text
P(EZ) = 1 - P(NEZ)
```

Task2 uses:

```text
1 = Engel I / success
0 = Engel II-IV / failure
```

Task2 does not consume ground-truth EZ labels or predicted EZ top-k pooling in
the main path.

## Main Flow

```powershell
python .\scripts\inspect_patient_records.py `
  --patient-records-pkl D:\nips-temp\pgc\patient_records.pkl

python .\scripts\build_physics_window_cache.py `
  --patient_records_pkl D:\nips-temp\pgc\patient_records.pkl `
  --output_cache D:\nips-temp\physics_cache\all_window_cache_physics_v1.pkl

python .\scripts\inspect_physics_cache.py `
  --cache-path D:\nips-temp\physics_cache\all_window_cache_physics_v1.pkl

python .\run_task1_pgc_ez.py `
  --window_cache_path D:\nips-temp\physics_cache\all_window_cache_physics_v1.pkl `
  --experiment_name T1_FULL_PGC `
  --output_dir D:\nips-temp\pgc\task1_full `
  --split_strategy 5fold `
  --n_splits 5

python .\run_task2_pgc_outcome.py `
  --window_cache_path D:\nips-temp\physics_cache\all_window_cache_physics_v1.pkl `
  --task1_checkpoint D:\nips-temp\pgc\task1_full\best_task1_backbone.pt `
  --experiment_name T2_FULL_ATTENTION_TOPOLOGY `
  --output_dir D:\nips-temp\pgc\task2_full `
  --split_strategy 5fold `
  --n_splits 5 `
  --freeze_backbone true
```

`scripts\build_patient_records_from_dre_nips.py` is copied from the existing
reader bridge. EDFs are read through provided local or mounted paths; this
project does not hardcode remote server credentials or SSH download logic.
