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

## Current v1 Feature Status

This repository currently implements PGC-SEEG v1.1 with:

- B0 self-referenced spectral/classical features
- physics-proxy v1 features
- TFCCM-lite nearest-neighbor cross-mapping graph
- Task1 EZ/NEZ binary localization
- Task2 Engel I vs II-IV binary outcome prediction
- fold-safe Task1-to-Task2 checkpoint loading

The current v1 causal graph is TFCCM-lite, not a full surrogate-tested TFCCM
implementation. The current v1 physics features are lightweight EDF-derived
proxies, not fully validated clinical biomarkers.

## v2 Strict Physics Mode

The cache builder also supports a v2 strict-physics cache:

```powershell
python .\scripts\build_physics_window_cache.py `
  --patient_records_pkl D:\nips-temp\pgc\patient_records.pkl `
  --output_cache D:\nips-temp\physics_cache\all_window_cache_physics_v2.pkl `
  --physics-mode strict `
  --line-noise-hz 50
```

Strict mode keeps the same model and dataset interface, but replaces the
6-dimensional proxy physics node features with 16 EDF-derived features:
robust aperiodic slope/offset fit quality, ripple/fast-ripple event summaries,
PAC vector-length features, and local synchrony. The cache metadata records
`physics_mode`, `physics_feature_level`, and `feature_names_physics`; inspect
the cache before training:

```powershell
python .\scripts\inspect_physics_cache.py `
  --cache-path D:\nips-temp\physics_cache\all_window_cache_physics_v2.pkl
```

Task1 and Task2 commands are unchanged except for `--window_cache_path`.

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
  --task1_checkpoint_dir D:\nips-temp\pgc\task1_full `
  --experiment_name T2_FULL_ATTENTION_TOPOLOGY `
  --output_dir D:\nips-temp\pgc\task2_full `
  --split_strategy 5fold `
  --n_splits 5 `
  --freeze_backbone true
```

Do not use a single cross-fold `best_task1_backbone.pt` for Task2 fold
evaluation. Task2 must load fold-specific Task1 checkpoints from
`--task1_checkpoint_dir`. A single `--task1_checkpoint` is allowed only with
`--allow_external_task1_checkpoint` for external-cohort pretraining or debug
runs, not for formal cross-validation.

Each Task1 run writes:

```text
output_dir/
  fold_0/best_task1_backbone.pt
  fold_0/split_subjects.json
  ...
  splits.json
```

`scripts\build_patient_records_from_dre_nips.py` is copied from the existing
reader bridge. EDFs are read through provided local or mounted paths; this
project does not hardcode remote server credentials or SSH download logic.
