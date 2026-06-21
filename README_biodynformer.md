# BioDynFormer Preictal-Only

This folder is an independent implementation for the preictal-only BioDynFormer
experiment line. It builds one reusable four-center feature bank, then runs v1,
v2, and final variants with patient-wise 5-fold and leave-one-center-out
validation.

Audit the metadata files first. This step checks outcome tables, quality
ratings, and whether signal files are actually available under the supplied
center roots:

```powershell
python scripts\audit_source_metadata.py `
  --metadata-dir C:\Users\gifoe\Downloads\all_seeg_data `
  --output-dir D:\nips-temp\biodynformer_source_audit `
  --lzu-root F:\兰大二院新SEEG数据 `
  --hup-root E:\DRE-nips\dataest `
  --multicenter-root E:\DRE-nips\dataest `
  --pediatric-root F:\儿科医院SEEG整理汇总
```

If the audit reports `can_build_feature_bank: false`, the metadata files were
parsed but EDF/NPY/NPZ signal files were not found. A preictal feature bank
cannot be built from quality/outcome spreadsheets alone.

Build a feature bank after the per-run signal paths are available:

```powershell
python scripts\build_feature_bank.py `
  --source four-center-raw `
  --centers lzu,hup,multicenter,pediatric `
  --lzu-manifest D:\path\to\lzu_manifest.csv `
  --hup-manifest D:\path\to\hup_manifest.csv `
  --multicenter-manifest D:\path\to\multicenter_manifest.csv `
  --pediatric-manifest D:\path\to\pediatric_manifest.csv `
  --output-dir D:\nips-temp\biodynformer_preictal_feature_bank `
  --quality-filter `
  --quality-keep-ratings GOOD,REVIEW `
  --quality-drop-ratings POOR `
  --strict-quality-reports
```

Each center manifest is a CSV with one row per seizure/run. Required columns:

```text
subject_id, run_id, signal_path, sfreq, seizure_onset_sec, channel_names, labels_ez, outcome, quality_rating
```

`signal_path` can point to `.npy`, `.npz`, `.csv`, `.txt`, or `.edf`. EDF reading
requires `mne`. `outcome` accepts `S/F`, `success/failure`, and `Engel I/II/III/IV`.

Recommended full pipeline command:

```powershell
python scripts\run_full_pipeline.py `
  --metadata-dir D:\all_seeg_data `
  --hup-participants-path E:\DRE-nips\dataest\participants.tsv `
  --lzu-root "F:\兰大二院新SEEG数据" `
  --hup-root "E:\DRE-nips\dataest" `
  --multicenter-root "E:\DRE-nips\dataest" `
  --pediatric-root "F:\儿科医院SEEG整理汇总" `
  --source-audit-output-dir D:\nips-temp\biodynformer_source_audit `
  --feature-bank-output-dir D:\nips-temp\biodynformer_preictal_feature_bank `
  --runs-output-dir D:\nips-temp\biodynformer_runs `
  --versions v1,v2,final `
  --tasks task1,task2 `
  --run-5fold `
  --run-loco `
  --resume
```

This command still requires per-center per-run manifests. If a center root does
not contain `manifest.csv`, pass it explicitly with `--lzu-manifest`,
`--hup-manifest`, `--multicenter-manifest`, and `--pediatric-manifest`.
Metadata audit outputs are center-level summaries and are not valid feature-bank
manifests.

Run all versions directly only after the feature bank exists:

```powershell
python scripts\run_all_versions.py `
  --feature-bank D:\nips-temp\biodynformer_preictal_feature_bank `
  --output-dir D:\nips-temp\biodynformer_runs `
  --versions v1,v2,final `
  --tasks task1,task2 `
  --run-5fold `
  --run-loco `
  --resume
```
