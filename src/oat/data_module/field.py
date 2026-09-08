"""Flat field value type: points with space-time coordinates and channel values."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

COORD_DIM = 3  # [x, y, t]


@dataclass
class Field:
    """A flat bundle of ``values`` at ``coords``.

    Shapes are either unbatched (``coords[N, D]``, ``values[N, C]``) or collated
    (``coords[B, N, D]``, ``values[B, N, C]``). ``coords``' last column is a numeric
    time ``t`` shared by all points of a sample; only the ``(x, y)`` columns are used
    for spatial distance. ``mask`` (if present) mirrors ``values``' shape and marks
    valid entries.
    """

    coords: Tensor
    values: Tensor
    var_names: list[str]
    mask: Tensor | None = None

    def __post_init__(self) -> None:
        if self.coords.shape[-1] != COORD_DIM:
            raise ValueError(
                f"coords last dim must be {COORD_DIM} ([x, y, t]), "
                f"got {self.coords.shape[-1]}"
            )
        if self.values.shape[-1] != len(self.var_names):
            raise ValueError(
                f"var_names has {len(self.var_names)} entries but values has "
                f"{self.values.shape[-1]} channels"
            )
        if self.coords.ndim not in (2, 3) or self.values.ndim != self.coords.ndim:
            raise ValueError(
                "coords and values must both be 2D (unbatched) or 3D (batched); "
                f"got {self.coords.ndim}d and {self.values.ndim}d"
            )
        if self.coords.shape[-2] != self.values.shape[-2]:
            raise ValueError(
                f"coords and values disagree on number of points: "
                f"{self.coords.shape[-2]} vs {self.values.shape[-2]}"
            )
        if self.values.ndim == 3 and self.coords.shape[0] != self.values.shape[0]:
            raise ValueError(
                f"coords and values disagree on batch size: "
                f"{self.coords.shape[0]} vs {self.values.shape[0]}"
            )
        if self.mask is not None and self.mask.shape != self.values.shape:
            raise ValueError(
                f"mask shape {tuple(self.mask.shape)} must equal values shape "
                f"{tuple(self.values.shape)}"
            )

    @property
    def spatial_coords(self) -> Tensor:
        """The ``(x, y)`` columns of ``coords``."""
        return self.coords[..., :2]

    @property
    def num_points(self) -> int:
        return self.values.shape[-2]

    @property
    def num_channels(self) -> int:
        return len(self.var_names)
