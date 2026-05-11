from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import warnings
import torch
import yaml

SUPPORT_RANK: dict[str, int] = {
    "positive": 0,
    "nonnegative": 1,
    "real": 2,
}


@dataclass
class VariableDef:
    name: str
    description: str
    unit: str
    support: str  # "real", "nonnegative", or "positive"
    default_distribution: str | None = None  # OATDistribution.name; obs_vars only

    def __post_init__(self) -> None:
        if self.support not in SUPPORT_RANK:
            raise ValueError(
                f"Unknown support '{self.support}' for variable '{self.name}'. "
                f"Must be one of: {list(SUPPORT_RANK)}"
            )


def load_catalog(path: str | Path) -> dict[str, VariableDef]:
    """Load variable catalog from a YAML file.
    Top-level key 'variables' maps variable names to dicts with keys:
    description, unit, support, and optionally default_distribution.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    catalog: dict[str, VariableDef] = {}
    for name, info in raw["variables"].items():
        catalog[name] = VariableDef(
            name=name,
            description=info["description"],
            unit=info["unit"],
            support=info["support"],
            default_distribution=info.get("default_distribution"),
        )
    return catalog


def constraint_to_support_str(constraint) -> str | None:
    """Classify a torch.distributions constraint to 'real', 'nonnegative', 'positive', or None.

    Probes the constraint with test tensor values (robust, no internal type names):
    - If -1.0 is valid → "real"
    - If -1.0 invalid but 0.0 valid → "nonnegative"
    - If both invalid but 1.0 valid → "positive"
    - Otherwise → None (unknown constraint, e.g. unit_interval, simplex)
    """
    try:
        neg = torch.tensor(-1.0)
        zero = torch.tensor(0.0)
        pos = torch.tensor(1.0)
        large = torch.tensor(2.0)
        if bool(constraint.check(neg)):
            return "real"
        if bool(constraint.check(zero)) and bool(constraint.check(large)):
            return "nonnegative"
        if bool(constraint.check(pos)) and bool(constraint.check(large)):
            return "positive"
        return None
    except Exception:
        return None


def check_distribution_compatibility(
    var_name: str,
    effective_support: str,
    dist_name: str,
    dist_cls: type,
) -> None:
    """Validate that dist_cls support is compatible with the variable's effective_support.

    After applying use_log=True to a positive variable, pass effective_support="real".

    Raises ValueError if variable support ⊄ distribution support
    (distribution cannot represent all values the variable can take).

    Issues UserWarning if distribution support ⊃ variable support
    (distribution can predict values the variable can never take).
    """
    var_rank = SUPPORT_RANK.get(effective_support)
    if var_rank is None:
        return

    dist_support_str = constraint_to_support_str(dist_cls.support)
    if dist_support_str is None:
        return

    dist_rank = SUPPORT_RANK[dist_support_str]

    if var_rank > dist_rank:
        raise ValueError(
            f"Variable '{var_name}' has support '{effective_support}' but distribution "
            f"'{dist_name}' has support '{dist_support_str}'. The distribution cannot "
            f"represent all values the variable can take. Use a distribution with broader "
            f"support or enable use_log=True if the variable has positive support."
        )
    if dist_rank > var_rank:
        warnings.warn(
            f"Variable '{var_name}' has support '{effective_support}' but distribution "
            f"'{dist_name}' has broader support '{dist_support_str}'. The distribution can "
            f"predict values outside the variable's natural range. Consider using a "
            f"distribution with '{effective_support}' support.",
            UserWarning,
            stacklevel=3,
        )
