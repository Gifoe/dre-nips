from __future__ import annotations

from typing import Any, Dict

import torch

from models.TeChEZ import Model as TeChEZModel


class Exp_Basic:
    def __init__(self, args: Any) -> None:
        self.args = args
        self.device = self._acquire_device()
        self.model_dict: Dict[str, Any] = {
            "TeChEZ": TeChEZModel,
        }

    def _acquire_device(self) -> torch.device:
        preferred = str(getattr(self.args, "device", "auto")).lower()
        if preferred == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if preferred.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(preferred)

    def _build_model(self):
        raise NotImplementedError
