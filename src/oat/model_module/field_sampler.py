"""Nearest-neighbour readout of flat Fields (the single interpolation primitive)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor

from oat.data_module.field import Field

Distance = Literal["euclidean"]


def _check_distance(distance: str) -> None:
    if distance != "euclidean":
        raise NotImplementedError(
            f"only distance='euclidean' is implemented; got {distance!r} "
            "(placeholder for a future metric)"
        )


def build_index(coords: Tensor, *, distance: Distance = "euclidean") -> cKDTree:
    """Build a cKDTree over the spatial ``(x, y)`` columns of ``coords``.

    ``coords`` is ``[N, D]`` or ``[B, N, D]``; for a batched, per-sample-identical
    static grid the tree is built from batch element 0.
    """
    _check_distance(distance)
    if coords.ndim == 3:
        coords = coords[0]
    xy = coords[:, :2].detach().cpu().numpy().astype(np.float64)
    return cKDTree(xy)


def nearest_read(
    field: Field,
    dst_coords: Tensor,
    *,
    index: cKDTree | None = None,
    distance: Distance = "euclidean",
) -> Tensor:
    """Read ``field.values`` at ``dst_coords`` by K=1 euclidean nearest neighbour.

    Distance uses only the spatial ``(x, y)`` columns; the ``t`` column is ignored.
    ``field.values`` is ``[N, C]`` or ``[B, N, C]``; ``dst_coords`` is ``[M, D]`` or
    ``[B, M, D]``. Returns ``[M, C]`` / ``[B, M, C]`` on ``field.values.device``.
    """
    _check_distance(distance)
    if index is None:
        index = build_index(field.coords, distance=distance)

    values = field.values
    device = values.device
    values_batched = values.ndim == 3
    dst_batched = dst_coords.ndim == 3

    def _query(xy: Tensor) -> Tensor:
        # xy: [M, 2] -> nearest source indices [M] as a LongTensor on `device`
        np_xy = xy[:, :2].detach().cpu().numpy().astype(np.float64)
        _, nn = index.query(np_xy)
        return torch.as_tensor(np.asarray(nn), dtype=torch.long, device=device)

    if not values_batched and not dst_batched:
        idx = _query(dst_coords)
        return values.index_select(0, idx)

    batch = values.shape[0] if values_batched else dst_coords.shape[0]
    outputs = []
    for b in range(batch):
        vals_b = values[b] if values_batched else values
        dst_b = dst_coords[b] if dst_batched else dst_coords
        idx = _query(dst_b)
        outputs.append(vals_b.index_select(0, idx))
    return torch.stack(outputs, dim=0)
