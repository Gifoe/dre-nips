# neuroez_c

`neuroez_c` implements B0-Pruned-EZBackbone for EZ/NEZ channel localization.
It trains from existing NeuroEZ window caches and does not use interictal,
physics, graph-message-passing, TCN/BiGRU, rank-loss, count-loss, or threshold
tuning branches.

Required cache fields:

```text
run_records
patient_index
window_features
window_relative_centers_sec
canonical_channels
labels
```

Run from the repository root:

```powershell
python .\run_neuroez_c.py --window_cache_path <all_window_cache.pkl> --output_dir <out>
```
