"""Data sources: one code path for nearest reads — point sets (cKDTree) and rasters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor


@dataclass
class SourceInfo:
    available_vars: list[str]
    native_coords: str | None
    has_mask: bool
    units: dict[str, str]


class Source(ABC):
    var_names: list[str]
    units: dict[str, str]
    has_mask: bool = False
    native_coord_source: str | None = None

    @abstractmethod
    def load(self, time_idx: int) -> Tensor:
        """All variables at the source's native points, shape [C, Npoints]."""

    def load_mask(self, time_idx: int) -> Tensor | None:
        """Validity mask [C, Npoints], or None when the source has no mask."""
        return None

    @abstractmethod
    def build_index(self, dst_coords: Tensor) -> Tensor:
        """Long tensor [M] mapping each destination point to a native point index."""

    def build_interp(self, dst_coords: Tensor, k: int) -> tuple[Tensor, Tensor]:
        """IDW k-NN read: long indices [M, k] and normalized weights [M, k]."""
        raise NotImplementedError

    @property
    def info(self) -> SourceInfo:
        return SourceInfo(
            available_vars=list(self.var_names),
            native_coords=self.native_coord_source,
            has_mask=self.has_mask,
            units=dict(self.units),
        )


class PointSetSource(Source):
    def __init__(
        self,
        coords: Tensor,
        var_names: list[str],
        values_fn: Callable[[int], Tensor],
        *,
        units: dict[str, str] | None = None,
        mask_fn: Callable[[int], Tensor] | None = None,
        native_coord_source: str | None = None,
    ) -> None:
        self.var_names = list(var_names)
        self.units = dict(units or {})
        self._values_fn = values_fn
        self._mask_fn = mask_fn
        self.has_mask = mask_fn is not None
        self.native_coord_source = native_coord_source
        self._tree = cKDTree(coords[..., :2].detach().cpu().numpy())

    def load(self, time_idx: int) -> Tensor:
        return self._values_fn(time_idx)

    def load_mask(self, time_idx: int) -> Tensor | None:
        return None if self._mask_fn is None else self._mask_fn(time_idx)

    def build_index(self, dst_coords: Tensor) -> Tensor:
        _, idx = self._tree.query(dst_coords[..., :2].detach().cpu().numpy())
        return torch.as_tensor(idx, dtype=torch.long)

    def build_interp(self, dst_coords: Tensor, k: int) -> tuple[Tensor, Tensor]:
        d, idx = self._tree.query(dst_coords[..., :2].detach().cpu().numpy(), k=k)
        d = d.reshape(-1, k)
        idx = idx.reshape(-1, k)
        w = 1.0 / np.maximum(d, 1e-12)
        w = w / w.sum(axis=-1, keepdims=True)
        return torch.as_tensor(idx, dtype=torch.long), torch.as_tensor(
            w, dtype=torch.float32
        )


class RasterSource(Source):
    def __init__(
        self,
        raster: Tensor,
        origin: tuple[float, float],
        res: tuple[float, float],
        var_names: list[str],
        *,
        units: dict[str, str] | None = None,
        native_coord_source: str | None = None,
    ) -> None:
        self.var_names = list(var_names)
        self.units = dict(units or {})
        self.native_coord_source = native_coord_source
        c, h, w = raster.shape
        self._flat = raster.reshape(c, h * w)
        self._h, self._w = h, w
        self._x0, self._y0 = origin
        self._rx, self._ry = res

    def load(self, time_idx: int) -> Tensor:
        return self._flat

    def build_index(self, dst_coords: Tensor) -> Tensor:
        xy = dst_coords[..., :2]
        col = (
            torch.round((xy[..., 0] - self._x0) / self._rx).long().clamp(0, self._w - 1)
        )
        row = (
            torch.round((xy[..., 1] - self._y0) / self._ry).long().clamp(0, self._h - 1)
        )
        return row * self._w + col

    def build_interp(self, dst_coords: Tensor, k: int) -> tuple[Tensor, Tensor]:
        xy = dst_coords[..., :2]
        x, y = xy[..., 0], xy[..., 1]
        fx = (x - self._x0) / self._rx
        fy = (y - self._y0) / self._ry
        col0 = torch.floor(fx).long().clamp(0, self._w - 1)
        col1 = (col0 + 1).clamp(0, self._w - 1)
        row0 = torch.floor(fy).long().clamp(0, self._h - 1)
        row1 = (row0 + 1).clamp(0, self._h - 1)

        cols = torch.stack([col0, col1, col0, col1], dim=-1)  # [M, 4]
        rows = torch.stack([row0, row0, row1, row1], dim=-1)  # [M, 4]
        flat = rows * self._w + cols  # [M, 4]

        cx = self._x0 + cols.to(xy.dtype) * self._rx
        cy = self._y0 + rows.to(xy.dtype) * self._ry
        dist = torch.sqrt((x.unsqueeze(-1) - cx) ** 2 + (y.unsqueeze(-1) - cy) ** 2)

        order = torch.argsort(dist, dim=-1)[:, :k]
        flat_k = torch.gather(flat, -1, order)
        dist_k = torch.gather(dist, -1, order)

        w = 1.0 / torch.clamp(dist_k, min=1e-12)
        w = w / w.sum(dim=-1, keepdim=True)
        return flat_k, w
