"""Weightless "predict the NWP value" reference baseline.

On the Field data path, interpolation already happened when the dataset
pre-sampled NWP at the query location; this baseline is just the resulting
per-channel map from NWP names to obs names, applied in raw data units.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class NWPInterpolatorConfig:
    nwp_names: list[str]
    obs_names: list[str]
    obs_to_nwp: dict[str, tuple[list[str], Callable[[dict[str, Tensor]], Tensor]]]


class NWPInterpolator(nn.Module):
    """Parameter-free NWP -> obs channel map.

    ``pred[..., j] = obs_to_nwp[obs_names[j]].fn(cols)`` where ``cols`` maps
    each name in ``nwp_names`` to its column of ``global_tokens``.
    """

    requires_observation: bool = False

    def __init__(self, config: NWPInterpolatorConfig) -> None:
        super().__init__()
        self.nwp_names = list(config.nwp_names)
        self.obs_names = list(config.obs_names)
        fns: list[Callable[[dict[str, Tensor]], Tensor]] = []
        for obs_name in self.obs_names:
            if obs_name not in config.obs_to_nwp:
                raise ValueError(f"no NWP mapping for obs var {obs_name!r}")
            deps, fn = config.obs_to_nwp[obs_name]
            missing = [d for d in deps if d not in self.nwp_names]
            if missing:
                raise ValueError(
                    f"NWP deps {missing} for {obs_name!r} not in nwp_names "
                    f"{self.nwp_names}"
                )
            fns.append(fn)
        self._fns = fns

    def forward(self, global_tokens: Tensor) -> Tensor:
        """``[..., C_nwp] -> [..., C_obs]`` (any number of leading dims)."""
        cols = {n: global_tokens[..., i] for i, n in enumerate(self.nwp_names)}
        return torch.stack([fn(cols) for fn in self._fns], dim=-1)
