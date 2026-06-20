# B0-Pruned-EZBackbone

This workspace contains a pruned EZ/NEZ channel localization model that reuses
existing NeuroEZ window cache files. It is not a surgery-outcome model and does
not re-extract EDF files.

## Cache Contract

Use an existing `*_window_cache.pkl` or `all_window_cache.pkl` with:

```text
run_records
patient_index
run_records[*].sample.window_features
run_records[*].sample.window_relative_centers_sec
patient_index[*].canonical_channels
patient_index[*].labels
```

`window_adjacency` may be present in old caches, but this model ignores it.
The expected cache feature order is the standard 20-dimensional window feature
schema: 13 spectral/classical features followed by 7 graph-node features. The
pruned model uses only the spectral/classical subset.

## Model

- self-comparison: `abs + delta + zdelta + ratio`
- features: delta, theta, beta, low-gamma, high-gamma, RMS, variance, line length, spectral entropy
- normalization: patient-relative channel z-score
- temporal aggregation: masked mean pooling
- channel interaction: lightweight channel attention
- cross-seizure aggregation: masked mean plus standard deviation
- training loss: masked channel-level BCE only

Removed paths include graph-node feature groups, adjacency message passing,
TCN/BiGRU/raw CNN, physics/interictal branches, supervised contrastive loss,
patient adversarial/disentanglement, rank loss, count loss, and threshold
tuning.

## Run

```powershell
python .\run_neuroez_c.py `
  --window_cache_path D:\nips-temp\neuroez_c_four_center_caches\all_window_cache.pkl `
  --output_dir D:\nips-temp\b0_pruned_results\manual
```

Audit a cache before training:

```powershell
python .\scripts\inspect_b0_ablation_cache.py --cache-path D:\nips-temp\neuroez_c_four_center_caches\all_window_cache.pkl
```
