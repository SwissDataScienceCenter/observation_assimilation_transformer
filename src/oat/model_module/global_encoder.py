import torch
from typing import Any, Dict
from oat.model_module.activations import get_activation
from oat.model_module.topographic_encoder import _FlattenedBatchNorm1d
from abc import ABC, abstractmethod


class BaseGlobalEncoder(torch.nn.Module, ABC):
    """Base class for global encoders processing global data inputs."""

    def __init__(self, **kwargs: Any) -> None:
        """
        Args:
            **kwargs: Additional configuration parameters for subclasses.
        """
        super().__init__()

    @abstractmethod
    def forward(
        self, global_data: Dict[str, Any], local_data: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Args:
            global_data: Dictionary of global data tensors.
            local_data: Should contain 'coords' key with local coordinates. [B, S, 2]
        Returns:
            torch.Tensor: Encoded global features at local positions [B, S, D].
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class PointGlobalEncoder(BaseGlobalEncoder):
    """Point-based global encoder: MLP over pre-sampled NWP tokens.

    Reads local_data['global_tokens'] [B, N, C_nwp] and returns [B, N, out].
    No grid_sample and no global_field.
    """

    def __init__(self, *, mlp_params: dict[str, Any], **kwargs: Any) -> None:
        super().__init__()
        in_dim = mlp_params["in_channels"]
        out_dim = mlp_params["out_channels"]
        hidden_dims = list(mlp_params.get("hidden_dims", []) or [])
        act_fn = mlp_params.get("act_fn", "GELU")
        use_layernorm = mlp_params.get("use_layernorm", False)
        use_batchnorm = mlp_params.get("use_batchnorm", False)
        layers: list[torch.nn.Module] = []
        d = in_dim
        for h in hidden_dims:
            layers.append(torch.nn.Linear(d, h))
            if use_layernorm:
                layers.append(torch.nn.LayerNorm(h))
            if use_batchnorm:
                layers.append(_FlattenedBatchNorm1d(h))
            layers.append(get_activation(act_fn))
            d = h
        layers.append(torch.nn.Linear(d, out_dim))
        self.model = torch.nn.Sequential(*layers)

    def forward(
        self, global_data: dict[str, Any], local_data: dict[str, Any]
    ) -> torch.Tensor:
        return self.model(local_data["global_tokens"])
