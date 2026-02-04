import torch
from typing import Any, Dict
from monai.networks import nets
from oat.data_module.wind_dataset import pos_transform
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
    
    
class UNetGlobalEncoder(BaseGlobalEncoder):
    """Global encoder using UNet architecture."""
    
    def __init__(self, *, unet_params: Dict[str, Any], field_extent: Dict[str, Any], **kwargs: Any) -> None:
        """
        Args:
            unet_params: Parameters for UNet model. See [monai](https://monai.readthedocs.io/en/stable/networks.html#unet) documentation for details.
            field_extent: Dictionary with 'min' and 'max', having the extent for all coordinates to allow for interpolation.
        """
        super().__init__()
        
        self.model = nets.UNet(
            **unet_params
        )
        self.field_extent = field_extent
        
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
    
    def interpolate(self, output: torch.Tensor, local_data: Dict[str, Any]) -> torch.Tensor:
        pos_transformed = pos_transform(local_data['coords'], 
                                self.field_extent['min'], 
                                self.field_extent['max']).unsqueeze(2)
        emb = F.grid_sample(output, pos_transformed, align_corners=True, mode='bilinear', padding_mode='border')
        emb = emb.squeeze(-1).permute(0, 2, 1)
        return emb
    
    
class SwinUNETRGlobalEncoder(BaseGlobalEncoder):
    """Global encoder using SwinUNETR architecture."""

    def __init__(self, *, swin_unetr_params: Dict[str, Any], field_extent: Dict[str, Any], **kwargs: Any) -> None:
        """
        Args:
            swin_unetr_params: Parameters for SwinUNETR model. See [monai](https://monai.readthedocs.io/en/stable/networks.html#swinunetr) documentation for details.
            field_extent: Dictionary with 'min' and 'max', having the extent for all coordinates to allow for interpolation.
        """
        super().__init__()
        
        self.model = nets.SwinUNETR(
            **swin_unetr_params
        )
        self.field_extent = field_extent
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
        pos_transformed = pos_transform(local_data['coords'], 
                                self.field_extent['min'], 
                                self.field_extent['max']).unsqueeze(2)
        emb = F.grid_sample(output, pos_transformed, align_corners=True, mode='bilinear', padding_mode='border')
        emb = emb.squeeze(-1).permute(0, 2, 1)
        return emb