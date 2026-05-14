from __future__ import annotations

from math import inf
from typing import NamedTuple

from torch.distributions import constraints


class _Bounds(NamedTuple):
    lower_bound: float = -inf
    upper_bound: float = inf
    left_open: bool = False
    right_open: bool = False


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
