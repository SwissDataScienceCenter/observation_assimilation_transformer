import torch
from typing import Any, Dict
from monai.networks import nets
from oat.data_module.wind_dataset import pos_transform
from oat.model_module.activations import get_activation
from torch.nn import functional as F
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
    def forward(self, global_data: Dict[str, Any], local_data: Dict[str, Any]) -> torch.Tensor:
        """
        Args:
            global_data: Dictionary of global data tensors.
            local_data: Should contain 'coords' key with local coordinates. [B, S, 2]
        Returns:
            torch.Tensor: Encoded global features at local positions [B, S, D].
        """
        raise NotImplementedError("Subclasses must implement the forward method.")
    
    
class InterpolationGlobalEncoder(BaseGlobalEncoder):
    
    def __init__(self, *, field_extent: Dict[str, Any], interpolation_params: Dict[str, Any] | None = None, **kwargs: Any) -> None:
        """
        Args:
            interpolation_params: Parameters for interpolation. Should contain 'mode' and 'padding_mode' keys for F.grid_sample. Defaults to bilinear and border respectively.
            field_extent: Dictionary with 'min' and 'max', having the extent for all coordinates to allow for interpolation.
        """
        super().__init__()
        self.register_buffer("field_min", torch.as_tensor(field_extent['min']))
        self.register_buffer("field_max", torch.as_tensor(field_extent['max']))
        if interpolation_params is None:
            interpolation_params = {}
        self._interpolation_params = interpolation_params
        
    def interpolate(self, output: torch.Tensor, local_data: Dict[str, Any]) -> torch.Tensor:
        pos_transformed = pos_transform(local_data['coords'], 
                                self.field_min, 
                                self.field_max).unsqueeze(2)
        emb = F.grid_sample(output, pos_transformed, 
                            align_corners=True, 
                            mode=self._interpolation_params.get('mode', 'bilinear'), 
                            padding_mode=self._interpolation_params.get('padding_mode', 'border'))
        emb = emb.squeeze(-1).permute(0, 2, 1)
        return emb
        
        
    def forward(self, global_data: Dict[str, Any], local_data: Dict[str, Any]) -> torch.Tensor:
        """
        Args:
            global_data: Dictionary of global data tensors. Should contain 'global_field' key with shape [B, C, H, W].
            local_data: Should contain 'coords' key with local coordinates. [B, S, 2]
        Returns:
            Encoded global features at local positions [B, S, C].
        """
        output = global_data["global_field"]
        interp_result = self.interpolate(output, local_data)
        return interp_result
    
class MLPGlobalEncoder(InterpolationGlobalEncoder):
    
    def __init__(self, *, mlp_params: Dict[str, Any], field_extent: Dict[str, Any], **kwargs: Any) -> None:
        """
        Args:
            mlp_params: Parameters for the MLP applied after interpolation. Should contain 'hidden_dims' key with list of hidden dimensions, 'in_channels' key for the input dimension of the first layer, 'out_channels' key for the output dimension, and 'act_fn' key for the activation function.
        """
        if 'out_channels' not in mlp_params:
            raise KeyError("mlp_params must contain 'out_channels' key for the output dimension.")
        if 'in_channels' not in mlp_params:
            raise KeyError("mlp_params must contain 'in_channels' key for the input dimension of the first layer.")
        super().__init__(interpolation_params=mlp_params.get('interpolation_params', None), field_extent=field_extent, **kwargs)
        hidden_dims = mlp_params.get('hidden_dims', [])
        input_dim = mlp_params['in_channels']  # Optional input dimension for the first layer, if different from interpolation output dimension
        output_dim = mlp_params['out_channels']
        
        layers = []
        for hidden_dim in hidden_dims:
            if input_dim is not None:
                layers.append(torch.nn.Linear(input_dim, hidden_dim))
                layers.append(get_activation(mlp_params.get('act_fn', 'ReLU')))
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, output_dim))
        self.model = torch.nn.Sequential(*layers)
        
    def forward(self, global_data: Dict[str, Any], local_data: Dict[str, Any]) -> torch.Tensor:
        emb = super().forward(global_data, local_data)
        emb = self.model(emb)
        return emb
    
class UNetGlobalEncoder(InterpolationGlobalEncoder):
    """Global encoder using UNet architecture."""
    
    def __init__(self, *, unet_params: Dict[str, Any], field_extent: Dict[str, Any], **kwargs: Any) -> None:
        """
        Args:
            unet_params: Parameters for UNet model. See [monai](https://monai.readthedocs.io/en/stable/networks.html#unet) documentation for details.
            field_extent: Dictionary with 'min' and 'max', having the extent for all coordinates to allow for interpolation.
        """
        super().__init__(field_extent=field_extent, interpolation_params=unet_params.get('interpolation_params', None), **kwargs)
        
        self.model = nets.UNet(
            **unet_params
        )
        
    def forward(self, global_data: Dict[str, Any], local_data: Dict[str, Any]) -> torch.Tensor:
        """
        Args:
            global_data: Dictionary of global data tensors.
            local_data: Should contain 'coords' key with local coordinates. [B, S, 2]
        Returns:
            Encoded global features at local positions [B, S, D].
        """
        output = self.apply_model(global_data)
        # interpolation
        emb = self.interpolate(output, local_data)
        return emb
    
    def apply_model(self, global_data: Dict[str, Any]) -> torch.Tensor:
        return self.model(global_data["global_field"])
    
    
class _ResBlock(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int, act_fn: str) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(channels)
        self.act = get_activation(act_fn)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(channels)
        self.act2 = get_activation(act_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + residual)


class ResNetCNNGlobalEncoder(InterpolationGlobalEncoder):
    """Global encoder using a stack of ResNet-style CNN blocks."""

    def __init__(self, *, resnet_cnn_params: Dict[str, Any], field_extent: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(field_extent=field_extent, interpolation_params=resnet_cnn_params.get('interpolation_params', None), **kwargs)
        in_channels = resnet_cnn_params['in_channels']
        out_channels = resnet_cnn_params['out_channels']
        num_blocks = resnet_cnn_params.get('num_blocks', 4)
        kernel_size = resnet_cnn_params.get('kernel_size', 3)
        act_fn = resnet_cnn_params.get('act_fn', 'ReLU')
        pad = kernel_size // 2
        layers: list[torch.nn.Module] = [
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            get_activation(act_fn),
        ]
        for _ in range(num_blocks):
            layers.append(_ResBlock(out_channels, kernel_size, act_fn))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, global_data: Dict[str, Any], local_data: Dict[str, Any]) -> torch.Tensor:
        output = self.model(global_data["global_field"])
        return self.interpolate(output, local_data)


class SwinUNETRGlobalEncoder(InterpolationGlobalEncoder):
    """Global encoder using SwinUNETR architecture."""

    def __init__(self, *, swin_unetr_params: Dict[str, Any], field_extent: Dict[str, Any], **kwargs: Any) -> None:
        """
        Args:
            swin_unetr_params: Parameters for SwinUNETR model. See [monai](https://monai.readthedocs.io/en/stable/networks.html#swinunetr) documentation for details.
            field_extent: Dictionary with 'min' and 'max', having the extent for all coordinates to allow for interpolation.
        """
        super().__init__(field_extent=field_extent, interpolation_params=swin_unetr_params.get('interpolation_params', None), **kwargs)
        
        self.model = nets.SwinUNETR(
            **swin_unetr_params
        )

        assert self.model.patch_size == 2, "This padding logic+monai only works for patch_size 2 currently."
        
    def _pad(self, img: torch.Tensor, padding_x: int, padding_y: int, unpad: bool=False) -> torch.Tensor:
        if not unpad:
            img_padded = F.pad(img, (padding_y // 2, 
                                     padding_y // 2 + (padding_y % 2), 
                                     padding_x // 2, 
                                     padding_x // 2 + (padding_x % 2)), mode='replicate')
            return img_padded
        else:
            return img[..., padding_x // 2: img.shape[-2] - padding_x // 2 - (padding_x % 2), 
                       padding_y // 2: img.shape[-1] - padding_y // 2 - (padding_y % 2)]
    

    def forward(self, global_data: Dict[str, Any], local_data: Dict[str, Any]) -> torch.Tensor:
        """
        Args:
            global_data: Dictionary of global data tensors.
            local_data: Should contain 'coords' key with local coordinates. [B, S, 2]
        Returns:
            Encoded global features at local positions [B, S, D].
        """
        global_field = global_data["global_field"]  # Expecting shape (B, C, H, W)
        padding_divisor = int(32)
        padding_x = (padding_divisor - global_field.shape[-2] % padding_divisor) % padding_divisor
        padding_y = (padding_divisor - global_field.shape[-1] % padding_divisor) % padding_divisor
        global_field_padded = self._pad(global_field, padding_x, padding_y, unpad=False)
        output = self.model(global_field_padded)
        output = self._pad(output, padding_x, padding_y, unpad=True)
        # interpolation
        # pos_transformed = pos_transform(local_data['coords'], 
        #                         self.field_min, 
        #                         self.field_max).unsqueeze(2)
        # emb = F.grid_sample(output, pos_transformed, align_corners=True, mode='bilinear', padding_mode='border')
        # emb = emb.squeeze(-1).permute(0, 2, 1)
        # interpolation
        emb = self.interpolate(output, local_data)
        return emb