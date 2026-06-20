from __future__ import annotations

__all__ = ["NeuroEZCModel"]


def __getattr__(name: str):
    if name == "NeuroEZCModel":
        from .model import NeuroEZCModel

        return NeuroEZCModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
