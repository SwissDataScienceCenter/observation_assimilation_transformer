from oat.model_module.activations import get_activation

from typing import Any

import torch
import torch.nn as nn


class TopographyEncoder(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: int):
        super().__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels


class _FlattenedBatchNorm1d(nn.Module):
    """BatchNorm1d over the channel (last) dim of an ``[..., C]`` tensor.

    ``nn.BatchNorm1d`` expects channels at dim 1; this flattens every leading
    dim into the batch dim first, so stats are pooled over all points in the
    forward pass (e.g. every station in every batch item), not just dim 0.
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        out: torch.Tensor = self.bn(x.reshape(-1, shape[-1])).reshape(shape)
        return out


class MLPTopographyEncoder(TopographyEncoder):
    """Point-wise MLP topo encoder: [..., C] -> [..., out_channels]."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_dims: list[int] | tuple[int, ...] | None = None,
        act_fn: str = "GELU",
        use_layernorm: bool = False,
        use_batchnorm: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        dims = list(hidden_dims or [])
        layers: list[nn.Module] = []
        d = in_channels
        for h in dims:
            layers.append(nn.Linear(d, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            if use_batchnorm:
                layers.append(_FlattenedBatchNorm1d(h))
            layers.append(get_activation(act_fn))
            d = h
        layers.append(nn.Linear(d, out_channels))
        self.model = nn.Sequential(*layers)

    def forward(self, topography: torch.Tensor) -> torch.Tensor:
        return self.model(topography)
