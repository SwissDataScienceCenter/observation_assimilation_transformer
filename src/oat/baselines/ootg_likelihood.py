from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import transform_to

from oat.model_module.distributions import NormalOAT, ZeroInflatedLogNormal


DIST_PARAM_COUNTS: dict[str, int] = {
    "normal": 2,                    # loc, scale
    "zero_inflated_lognormal": 3,   # gate, loc, scale
}

_DIST_CLASSES = {
    "normal": NormalOAT,
    "zero_inflated_lognormal": ZeroInflatedLogNormal,
}


def decoder_out_dim(obs_variable_to_distribution: dict[str, str]) -> int:
    """Total number of raw decoder outputs for the given variable→distribution map."""
    return sum(DIST_PARAM_COUNTS[d] for d in obs_variable_to_distribution.values())


class JointMixedDistribution:
    """Holds one distribution per output variable; treats them as independent.

    log_prob, mean, and sample operate elementwise across variables and return
    tensors with a trailing dim_y dimension, consistent with OOTGBatch.yt shape.
    """

    def __init__(self, dists: list) -> None:
        self.dists = dists  # list of Distribution, one per variable

    def log_prob(self, y: torch.Tensor) -> torch.Tensor:
        """Per-variable log probabilities.

        Args:
            y: (..., dim_y) tensor, variables in the same order as self.dists

        Returns:
            (..., dim_y) log probabilities
        """
        return torch.stack(
            [d.log_prob(y[..., i]) for i, d in enumerate(self.dists)], dim=-1
        )

    @property
    def mean(self) -> torch.Tensor:
        """(..., dim_y) tensor of per-variable distribution means."""
        return torch.stack([d.mean for d in self.dists], dim=-1)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Sample each variable independently; returns (..., dim_y)."""
        return torch.stack(
            [d.sample(sample_shape) for d in self.dists], dim=-1
        )


class MixedVariableLikelihood(nn.Module):
    """Maps raw decoder output to a JointMixedDistribution.

    Splits the decoder's last dimension into per-variable parameter slices and
    applies the appropriate distribution class (Normal or ZeroInflatedLogNormal).
    Parameters are constrained via transform_to (same pattern as OAT's DistributionMap).

    Args:
        obs_variable_to_distribution: ordered dict mapping variable name → dist name,
            e.g. {"temperature": "normal", "precipitation": "zero_inflated_lognormal"}.
            The order determines which slice of the decoder output maps to which variable.
    """

    def __init__(self, obs_variable_to_distribution: dict[str, str]) -> None:
        super().__init__()
        for v, d in obs_variable_to_distribution.items():
            if d not in _DIST_CLASSES:
                raise ValueError(
                    f"Unknown distribution {d!r} for variable {v!r}. "
                    f"Supported: {list(_DIST_CLASSES)}"
                )
        self.var_names = list(obs_variable_to_distribution.keys())
        self.dist_names = list(obs_variable_to_distribution.values())
        self.dist_classes = [_DIST_CLASSES[d] for d in self.dist_names]
        self.param_counts = [DIST_PARAM_COUNTS[d] for d in self.dist_names]

    def forward(self, x: torch.Tensor) -> JointMixedDistribution:
        """
        Args:
            x: (..., sum(param_counts)) raw decoder output

        Returns:
            JointMixedDistribution with one distribution per variable
        """
        dists = []
        offset = 0
        for dist_class, n in zip(self.dist_classes, self.param_counts):
            raw = x[..., offset : offset + n]
            params = {}
            for i, (pname, constraint) in enumerate(
                dist_class.arg_constraints.items()
            ):
                params[pname] = transform_to(constraint)(raw[..., i])
            dists.append(dist_class(**params))
            offset += n
        return JointMixedDistribution(dists)
