from oat.model_module.activations import get_activation

from abc import ABC, abstractmethod
from typing import Any

import math
import torch
import torch.nn as nn
from einops import rearrange



class TopographyEncoder(nn.Module, ABC):
    def __init__(self, *, in_channels: int, out_channels: int):
        super().__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels


class CNNTopographyEncoder(TopographyEncoder):

    def __init__(self, *,
                 in_channels: int,
                 out_channels: int,
                 n_layers: int,
                 kernel_sizes: int | list[int],
                 initial_hw: tuple[int, int] | None = None,
                 target_hw: list[tuple[int, int]] | None = None,
                 channels: list[int] | None = None,
                 act_fn: str = "ReLU",
                 act_fn_kwargs: dict[str, Any] | None = None,
                 use_batchnorm: bool = True
                 ):
        """ Encoder for topographic patches of shape [C,H,W] or [B,C,H,W] with a CNN 
        backbone. Returns a flat vector of shape [B, out_channels] or [1, out_channels].

        Args:
            in_channels: The input channels of the patch.
            out_channels: The number of output embedding feature.
            n_layers (int): The number of convolutional layers.
            kernel_sizes: The size of the learnable kernels.
                If integer, the same size is used for all layers.
            initial_hw: The initial height and width of an input sample.
                Must be given is target_hw is set to None.
                In that case, the height and width are halved at each layer.
            target_hw: The intermediate output sizes [H,W]
                after each layer. If not given, the height and width are halved at each layer.
            channels: The number of learnable filters for each layer. If not given, the 
                channels are doubled at each layer.
            act_fn: The activation string of an activation function of torch.nn
                such as ReLU, GELU (case sensitive). The activation is used after each
                convolution (but before batch norm).
            act_fn_kwargs: Any key-value arguments necessary to initialize the activation
                function layer.
            use_batchnorm: Whether a batch norm is applied after every
                convolution.
        """
        super().__init__(in_channels=in_channels, out_channels=out_channels)

        # TODO add kwargs for some act (eg LeakyReLU)
        if target_hw is None:
            assert initial_hw is not None, "With initial_hw set to None, \
            you must define the initial height and width."

            # Create targets hw that are halved at each layer
            target_hw = [
                (math.ceil(initial_hw[0] / 2**i), math.ceil(initial_hw[1] / 2**i))
                for i in range(n_layers)
            ]

        if channels is None:
            # Double the number of channels for each layer
            channels = [in_channels * 2**i for i in range(n_layers)]

        assert len(target_hw) == n_layers
        assert len(channels) == n_layers

        act_fn_kwargs = act_fn_kwargs or {}

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_layers
        else:
            assert len(kernel_sizes) == n_layers

        curr_c = in_channels
        blocks = []

        # Create each conv layer (+ optional BN)
        for i in range(n_layers):
            ks = kernel_sizes[i]

            layers = [
                nn.Conv2d(curr_c, channels[i], kernel_size=ks)
            ]

            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels[i]))

            layers.append(get_activation(act_fn, **act_fn_kwargs))
            layers.append(nn.AdaptiveAvgPool2d(target_hw[i]))

            blocks.append(nn.Sequential(*layers))
            curr_c = channels[i]

        self.conv_blocks = nn.Sequential(*blocks)

        # Create output head by flattening the last feature map
        # TODO simple linear projection is fine?
        final_hw = target_hw[-1]
        flat_dim = curr_c * final_hw[0] * final_hw[1]
        self.output_head = nn.Linear(flat_dim, out_channels)

    def forward(self, topography_patch: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the topography encoder.
        
        Args:
            topography_patch: The input topography patch of shape [..., C, H, W].

        Returns:
            The encoded topography features as a tensor.
        """

        leading_dims = topography_patch.shape[:-3]
        C, H, W = topography_patch.shape[-3:]
        out = self.conv_blocks(topography_patch.reshape(-1, C, H, W))
        if leading_dims:
            out_flat = out.reshape(*leading_dims, -1)
        else:
            # Preserve a batch dimension for 3D inputs [C, H, W] -> [1, out_channels]
            out_flat = out.reshape(1, -1)
        out_flat = self.output_head(out_flat)
        return out_flat
