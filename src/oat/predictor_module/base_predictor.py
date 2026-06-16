from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import pytorch_lightning as pl


class BasePredictor(pl.LightningModule, ABC):
    def __init__(self, output_var_names: list[str]) -> None:
        super().__init__()
        self.output_var_names: list[str] = output_var_names
        self.var_to_indx: dict[str, int] = {v: i for i, v in enumerate(output_var_names)}
        self.indx_to_var: dict[int, str] = {i: v for i, v in enumerate(output_var_names)}

    @abstractmethod
    def _normalize_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        ...
