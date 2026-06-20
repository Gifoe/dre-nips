from __future__ import annotations

DEFAULT_WINDOW_CACHE = r"D:\nips-temp\neuroez_c_four_center_caches\all_window_cache.pkl"


def apply_pruned_defaults(args) -> None:
    setattr(args, "model_family", "b0_pruned_ez_backbone")
    setattr(args, "positive_label", "nez")
    setattr(args, "score_semantics", "nez_probability")
    if not getattr(args, "window_cache_path", None):
        setattr(args, "window_cache_path", DEFAULT_WINDOW_CACHE)
    if not getattr(args, "b0_feature_parts", None):
        setattr(args, "b0_feature_parts", "abs,delta,zdelta,ratio")
    if not getattr(args, "b0_feature_groups", None):
        setattr(args, "b0_feature_groups", "spectral_classical")
    setattr(args, "drop_high_ez_fraction_lzu", bool(getattr(args, "drop_high_ez_fraction_lzu", True)))
    setattr(args, "lzu_max_ez_fraction", float(getattr(args, "lzu_max_ez_fraction", 0.40)))


__all__ = ["DEFAULT_WINDOW_CACHE", "apply_pruned_defaults"]
