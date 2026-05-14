from __future__ import annotations

from math import inf
from typing import NamedTuple

import torch
from torch.distributions import constraints


class _Bounds(NamedTuple):
    lower_bound: float = -inf
    upper_bound: float = inf
    lower_open: bool = False
    upper_open: bool = False


def _to_bounds(c: constraints.Constraint) -> _Bounds:
    if getattr(c, "is_discrete", False):
        raise ValueError("Only continuous constraints are supported.")

    if c is constraints.real:
        return _Bounds(-inf, inf, True, True)
    if c is constraints.nonnegative:
        return _Bounds(0.0, inf, False, True)
    if c is constraints.positive:
        return _Bounds(0.0, inf, True, True)

    if isinstance(c, constraints._HalfOpenInterval):
        return _Bounds(float(c.lower_bound), float(c.upper_bound), False, True)

    if isinstance(c, constraints._Interval):
        return _Bounds(float(c.lower_bound), float(c.upper_bound), False, False)

    if isinstance(c, constraints._GreaterThanEq):
        return _Bounds(float(c.lower_bound), inf, False, True)
    if isinstance(c, constraints._GreaterThan):
        return _Bounds(float(c.lower_bound), inf, True, True)
    if isinstance(c, constraints._LessThan):
        return _Bounds(-inf, float(c.upper_bound), True, True)

    raise ValueError(f"Unsupported continuous constraint type: {type(c).__name__}")


class VariableSupport(constraints.Constraint):
    """Per-variable interval support for a multi-dimensional tensor.

    Parameters
    ----------
    lower_bounds:
        1-D tensor of shape ``(V,)`` — lower bound for each variable.
    upper_bounds:
        1-D tensor of shape ``(V,)`` — upper bound for each variable.
    dim:
        Dimension of the *variable* axis in tensors passed to ``check``.
        Default ``-1`` (last dimension).
    lower_open:
        Per-variable flag for strict lower bound (``value > lower_bound``).
        Pass a bool tensor of shape ``(V,)`` or a single ``bool`` applied to
        all variables.  Default ``False`` (closed lower bound).
    upper_open:
        Per-variable flag for strict upper bound (``value < upper_bound``).
        Same broadcast rules as ``lower_open``.  Default ``False``.
    """

    lower_bounds: torch.Tensor
    upper_bounds: torch.Tensor
    lower_open: torch.Tensor
    upper_open: torch.Tensor
    dim: int

    def __init__(
        self,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
        *,
        dim: int = -1,
        lower_open: torch.Tensor | bool = False,
        upper_open: torch.Tensor | bool = False,
    ) -> None:
        if lower_bounds.ndim != 1 or upper_bounds.ndim != 1:
            raise ValueError("lower_bounds and upper_bounds must be 1-D tensors.")
        n = lower_bounds.shape[0]
        if upper_bounds.shape[0] != n:
            raise ValueError("lower_bounds and upper_bounds must have the same length.")
        if not (lower_bounds < upper_bounds).all():
            raise ValueError(
                "lower_bounds must be strictly less than upper_bounds "
                "for all variables."
            )

        def _to_bool_tensor(flag: torch.Tensor | bool) -> torch.Tensor:
            if isinstance(flag, bool):
                return torch.full((n,), flag, dtype=torch.bool)
            t = flag.bool()
            if t.shape != (n,):
                raise ValueError(
                    f"open flags must have shape ({n},), got {tuple(t.shape)}"
                )
            return t

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.dim = dim
        # Infinite bounds are semantically open: force the flag regardless of input.
        self.lower_open = _to_bool_tensor(lower_open) | torch.isneginf(lower_bounds)
        self.upper_open = _to_bool_tensor(upper_open) | torch.isposinf(upper_bounds)

    def check(self, value: torch.Tensor) -> torch.Tensor:
        """Return a bool tensor of the same shape as *value*.

        Each slice along ``self.dim`` is checked against the corresponding
        variable's bounds.
        """
        ndim = value.ndim
        d = self.dim % ndim  # normalise to positive index

        # Reshape 1-D bound/flag tensors to broadcast over all other dims.
        shape = [1] * ndim
        shape[d] = -1

        lo = self.lower_bounds.reshape(shape)
        hi = self.upper_bounds.reshape(shape)
        lo_open = self.lower_open.reshape(shape)
        hi_open = self.upper_open.reshape(shape)

        lo_ok = torch.where(lo_open, value > lo, value >= lo)
        hi_ok = torch.where(hi_open, value < hi, value <= hi)
        return lo_ok & hi_ok

    def is_subset(self, other: constraints.Constraint, idx: int) -> bool:
        """Return True if variable *idx*'s interval is a subset of *other*.

        Converts *other* to a ``_Bounds`` via ``_to_bounds`` and compares
        bounds and openness flags numerically.  ``other.check()`` is never
        called.

        Parameters
        ----------
        other:
            Any ``torch.distributions.constraints.Constraint`` to compare
            against.
        idx:
            Index into the variable axis selecting which variable to test.
        """
        target = _to_bounds(other)
        lo = float(self.lower_bounds[idx].item())
        hi = float(self.upper_bounds[idx].item())
        lo_open = bool(self.lower_open[idx].item())
        hi_open = bool(self.upper_open[idx].item())

        # Lower bound: equality is fine when self excludes the point (open)
        # or target includes it (closed). Otherwise must be strictly inside.
        if lo_open or not target.lower_open:
            lower_ok = lo >= target.lower_bound
        else:
            lower_ok = lo > target.lower_bound

        if hi_open or not target.upper_open:
            upper_ok = hi <= target.upper_bound
        else:
            upper_ok = hi < target.upper_bound

        return lower_ok and upper_ok
