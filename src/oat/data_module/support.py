from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any

import torch
from torch.distributions import constraints


@dataclass(frozen=True)
class IntervalSupport(constraints.Constraint):
    lower_bound: float = -inf
    upper_bound: float = inf
    left_open: bool = False
    right_open: bool = False

    def check(self, value: torch.Tensor) -> torch.Tensor:
        if self.left_open:
            left_ok = value > self.lower_bound
        else:
            left_ok = value >= self.lower_bound

        if self.right_open:
            right_ok = value < self.upper_bound
        else:
            right_ok = value <= self.upper_bound

        return left_ok & right_ok

    def is_subset(self, other: constraints.Constraint) -> bool:
        target = _to_interval_support(other)

        lower_ok = (
            self.lower_bound > target.lower_bound
            or (
                self.lower_bound == target.lower_bound
                and (self.left_open or not target.left_open)
            )
        )
        upper_ok = (
            self.upper_bound < target.upper_bound
            or (
                self.upper_bound == target.upper_bound
                and (self.right_open or not target.right_open)
            )
        )
        return lower_ok and upper_ok


CONSTRAINT_REGISTRY: dict[str, type[IntervalSupport]] = {
    "interval": IntervalSupport,
    # alias accepted when full class name is passed as string
    "intervalsupport": IntervalSupport,
}


def create_variable_support(
    spec: str | constraints.Constraint,
    **kwargs: Any,
) -> IntervalSupport:
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key not in CONSTRAINT_REGISTRY:
            raise ValueError(f"Unknown variable support: {spec}")
        return CONSTRAINT_REGISTRY[key](**kwargs)

    return _to_interval_support(spec)


def _to_interval_support(c: constraints.Constraint) -> IntervalSupport:
    if isinstance(c, IntervalSupport):
        return c

    if getattr(c, "is_discrete", False):
        raise ValueError("Only continuous constraints are supported.")

    if c is constraints.real:
        return IntervalSupport()
    if c is constraints.nonnegative:
        return IntervalSupport(lower_bound=0.0)
    if c is constraints.positive:
        return IntervalSupport(lower_bound=0.0, left_open=True)

    if isinstance(c, constraints._HalfOpenInterval):
        return IntervalSupport(
            lower_bound=float(c.lower_bound),
            upper_bound=float(c.upper_bound),
            left_open=False,
            right_open=True,
        )

    if isinstance(c, constraints._Interval):
        return IntervalSupport(
            lower_bound=float(c.lower_bound),
            upper_bound=float(c.upper_bound),
        )

    if isinstance(c, constraints._GreaterThanEq):
        return IntervalSupport(lower_bound=float(c.lower_bound))
    if isinstance(c, constraints._GreaterThan):
        return IntervalSupport(lower_bound=float(c.lower_bound), left_open=True)
    if isinstance(c, constraints._LessThan):
        return IntervalSupport(upper_bound=float(c.upper_bound), right_open=True)

    raise ValueError(f"Unsupported continuous constraint type: {type(c).__name__}")
