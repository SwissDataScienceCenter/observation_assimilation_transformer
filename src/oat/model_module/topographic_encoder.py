from abc import ABC, abstractmethod

import torch
import torch.nn as nn

# TODO move in a shared location?
def get_activation(name: str, **kwargs):
    cls = getattr(nn, name, None)
    if cls is None:
        raise ValueError(f"Activation {name} not found in torch.nn")
    return cls(**kwargs)


class TopographyEncoder(nn.Module, ABC):
    def __init__(self, *, in_channels: int, out_channels: int):
        super().__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, topography_patch: torch.Tensor) -> torch.Tensor:
        return self._encode_patch(topography_patch)

    @abstractmethod
    def _encode_patch(self, topography_patch: torch.Tensor) -> torch.Tensor:
        """Encode a topography patch into a feature representation."""
        pass


class CNNTopographyEncoder(TopographyEncoder):

    def __init__(self, *,
                 in_channels: int,
                 out_channels: int,
                 kernel_sizes: int | list[int],
                 target_hw: list[tuple[int, int]],
                 channels: list[int],
                 act_fn: str,
                 use_batchnorm: bool = True
                 ):
        """ Encoder for topographic patches of shape [C,H,W] or [B,C,H,W] with a CNN 
        backbone. Returns a flat vector of shape [B, out_channels] or [1, out_channels].

        Args:
            in_channels (int): The input channels of the patch.
            out_channels (int): The number of output embedding feature.
            kernel_sizes (int | list[int]): The size of the learnable kernels.
                If integer, the same size is used for all layers.
            target_hw (list[tuple[int, int]]): The intermediate output sizes [H,W] 
                after each layer.
            channels (list[int]): The number of learnable filters for each layer.
                The length of this list defines the number of layers and must match
                the length of target_hw.
            act_fn (str): The activation string of an activation function of torch.nn
                such as ReLU, GELU (case sensitive). The activation is used after each
                convolution (but before batch norm).
            use_batchnorm (bool, optional): Whether a batch norm is applied after every
                convolution. Defaults to True.
        """
        super().__init__(in_channels=in_channels, out_channels=out_channels)

        # TODO add kwargs for some act (eg LeakyReLU)
        n_layers = len(channels)
        assert len(target_hw) == n_layers

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_layers
        else:
            assert len(kernel_sizes) == n_layers

        curr_c = in_channels
        blocks = []

        # Create each conv layer (+ optional BN)
        for i in range(n_layers):
            ks = kernel_sizes[i]
            padding = ks // 2

            layers = [
                nn.Conv2d(curr_c, channels[i], 
                          kernel_size=ks, padding=padding, padding_mode='replicate')
            ]
            # TODO 'replicate' ok to hardcode?

            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels[i]))

            layers.append(get_activation(act_fn))
            layers.append(nn.AdaptiveAvgPool2d(target_hw[i]))

            blocks.append(nn.Sequential(*layers))
            curr_c = channels[i]

        self.conv_blocks = nn.Sequential(*blocks)

        # Create output head by flattening the last feature map
        # TODO simple linear projection is fine?
        final_hw = target_hw[-1]
        flat_dim = curr_c * final_hw[0] * final_hw[1]
        self.flatten = nn.Flatten()
        self.output_head = nn.Linear(flat_dim, out_channels)

    def _encode_patch(self, topography_patch: torch.Tensor) -> torch.Tensor:

        if topography_patch.ndim == 3:
            # if the tensor is [C,H,W], add a dummy batch size
            # This is needed for batch norm 2d to work.
            topography_patch = topography_patch.unsqueeze(0)

        out = self.conv_blocks(topography_patch)
        out_flat = self.flatten(out)

        return self.output_head(out_flat)
