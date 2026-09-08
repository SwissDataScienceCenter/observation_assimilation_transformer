from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
import pytorch_lightning as pl

from oat.data_module.field import Field


class BasePredictor(pl.LightningModule, ABC):
    def __init__(
        self, output_var_names: list[str], field_roles: dict | None = None
    ) -> None:
        super().__init__()
        self.output_var_names: list[str] = output_var_names
        self.var_to_indx: dict[str, int] = {
            v: i for i, v in enumerate(output_var_names)
        }
        self.indx_to_var: dict[int, str] = {
            i: v for i, v in enumerate(output_var_names)
        }
        self.field_roles = field_roles
        self.project_index_coords: Callable[[torch.Tensor], torch.Tensor] | None = None

    @abstractmethod
    def _normalize_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        ...

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        if self.field_roles is None:
            return super().transfer_batch_to_device(batch, device, dataloader_idx)

        def mv(f: Field) -> Field:
            return Field(
                f.coords.to(device),
                f.values.to(device),
                f.var_names,
                f.mask.to(device) if f.mask is not None else None,
            )

        out = {}
        for k, v in batch.items():
            if isinstance(v, dict) and "fields" in v:
                out[k] = {"fields": {n: mv(f) for n, f in v["fields"].items()}}
            elif torch.is_tensor(v):
                out[k] = v.to(device)
            else:
                out[k] = v
        return out

    def set_index_projection(self, dataset: Any) -> None:
        """Resolve and store the dataset's index-coordinate projection.

        Stored as a plain attribute (not a hyperparameter) so it is never
        pickled into the checkpoint; the eval path re-establishes it from the
        rebuilt dataset.
        """
        from oat.predictor_module.field_batch_utils import resolve_index_projection

        self.project_index_coords = resolve_index_projection(dataset)
