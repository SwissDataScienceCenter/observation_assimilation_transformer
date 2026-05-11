from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
import warnings
import torch
from torch.distributions import Distribution, constraints
import yaml  # type: ignore[import-untyped]

SUPPORT_REGISTRY: dict[str, Any] = {
    "positive": constraints.positive,
    "nonnegative": constraints.nonnegative,
    "real": constraints.real,
    "interval": constraints.interval,
    "less_than": constraints.less_than,
    "greater_than": constraints.greater_than,
    "half_open_interval": constraints.half_open_interval,
}


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def get_bounds(support: constraints.Constraint) -> tuple[float, float]:
    lb = _to_float(support.lower_bound) if hasattr(support, "lower_bound") else float("-inf")
    ub = _to_float(support.upper_bound) if hasattr(support, "upper_bound") else float("inf")
    return lb, ub


def _includes_value(support: constraints.Constraint, value: float) -> bool:
    return bool(support.check(torch.tensor(value)))  # type: ignore[no-untyped-call]


def check_support_overlap(support1: constraints.Constraint, support2: constraints.Constraint) -> str:
    """Return overlap relationship of support1 with respect to support2.

    Returns:
    - "full" if support1 is a subset of support2.
    - "partial" if supports intersect but support1 is not fully contained in support2.
    - "none" if supports are disjoint.
    """
    if support1.is_discrete and support2.is_discrete:
        raise NotImplementedError("Support overlap check for discrete distributions is not implemented.")

    lb1, ub1 = get_bounds(support1)
    lb2, ub2 = get_bounds(support2)

    # Disjoint intervals: left/right separated, including open-boundary corner cases.
    if ub1 < lb2:
        return "none"
    if ub2 < lb1:
        return "none"
    if ub1 == lb2 and not (_includes_value(support1, ub1) and _includes_value(support2, lb2)):
        return "none"
    if ub2 == lb1 and not (_includes_value(support2, ub2) and _includes_value(support1, lb1)):
        return "none"

    # Containment check: support1 subset support2.
    lower_ok = False
    if lb2 == float("-inf"):
        lower_ok = True
    elif lb1 > lb2:
        lower_ok = True
    elif lb1 == lb2:
        # If support1 includes the shared lower boundary, support2 must include it too.
        lower_ok = (not _includes_value(support1, lb1)) or _includes_value(support2, lb2)

    upper_ok = False
    if ub2 == float("inf"):
        upper_ok = True
    elif ub1 < ub2:
        upper_ok = True
    elif ub1 == ub2:
        # If support1 includes the shared upper boundary, support2 must include it too.
        upper_ok = (not _includes_value(support1, ub1)) or _includes_value(support2, ub2)

    if lower_ok and upper_ok:
        return "full"
    return "partial"


def _build_constraint_from_spec(spec: Any) -> constraints.Constraint:
    if isinstance(spec, str):
        factory = SUPPORT_REGISTRY.get(spec)
        if factory is None:
            raise ValueError(f"Unknown support '{spec}'. Allowed: {list(SUPPORT_REGISTRY)}")
        if isinstance(factory, constraints.Constraint):
            return factory
        raise ValueError(
            f"Support '{spec}' requires parameters and must be configured as a mapping "
            f"with keys like 'name', 'lower_bound', 'upper_bound'."
        )

    if isinstance(spec, dict):
        name = spec.get("name")
        if not isinstance(name, str):
            raise ValueError("Support mapping must include string field 'name'.")

        if name not in SUPPORT_REGISTRY:
            raise ValueError(f"Unknown support '{name}'. Allowed: {list(SUPPORT_REGISTRY)}")

        if name == "interval":
            return constraints.interval(  # type: ignore[no-untyped-call]
                float(spec["lower_bound"]),
                float(spec["upper_bound"]),
            )
        if name == "half_open_interval":
            return constraints.half_open_interval(  # type: ignore[no-untyped-call]
                float(spec["lower_bound"]),
                float(spec["upper_bound"]),
            )
        if name == "less_than":
            return constraints.less_than(float(spec["upper_bound"]))  # type: ignore[no-untyped-call]
        if name == "greater_than":
            return constraints.greater_than(float(spec["lower_bound"]))  # type: ignore[no-untyped-call]

        factory = SUPPORT_REGISTRY[name]
        if isinstance(factory, constraints.Constraint):
            return factory
        raise ValueError(f"Support '{name}' unexpectedly requires parameters.")

    raise TypeError("Support spec must be a string or mapping.")

@dataclass
class VariableDef:
    name: str
    description: str
    unit: str
    support: constraints.Constraint
    default_distribution: str | None = None  # OATDistribution.name; obs_vars only

    def __post_init__(self) -> None:
        if not isinstance(self.support, constraints.Constraint):
            raise TypeError(
                f"support for variable '{self.name}' must be a torch Constraint, got "
                f"{type(self.support).__name__}"
            )


def load_catalog(path: str | Path) -> dict[str, VariableDef]:
    """Load variable catalog from a YAML file with a top-level 'variables' key."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    variables = raw.get("variables")
    if not isinstance(variables, dict):
        raise ValueError("Catalog YAML must contain a top-level 'variables' mapping.")

    catalog: dict[str, VariableDef] = {}
    for name, info in variables.items():
        catalog[name] = VariableDef(
            name=name,
            description=info["description"],
            unit=info["unit"],
            support=_build_constraint_from_spec(info["support"]),
            default_distribution=info.get("default_distribution"),
        )
    return catalog


def check_distribution_compatibility(
    var_name: str,
    effective_support: constraints.Constraint,
    dist_name: str,
    dist_cls: type[Distribution],
) -> None:
    """Raise ValueError for disjoint supports and warn on partial overlap."""
    dist_support = cast(constraints.Constraint, dist_cls.support)
    overlap = check_support_overlap(effective_support, dist_support)

    if overlap == "none":
        raise ValueError(
            f"Variable '{var_name}' support is not fully supported by distribution '{dist_name}' support. "
            "The distribution cannot represent the variable domain."
        )
    if overlap == "partial":
        warnings.warn(
            f"Variable '{var_name}' support partially overlaps with distribution '{dist_name}' support. "
            "Predictions may fall outside the variable's natural range or fail to cover it completely.",
            UserWarning,
            stacklevel=3,
        )
