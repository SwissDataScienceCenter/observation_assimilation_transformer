"""Nearest-neighbour readout of flat Fields (the single interpolation primitive)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.spatial import cKDTree
from torch import Tensor

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
