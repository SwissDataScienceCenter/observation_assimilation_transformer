import torch
from typing import Any, Dict
from monai.networks.nets import SwinUNETR
from monai.networks.layers import split_args, Pad
from torch.nn.functional import grid_sample
from oat.data_module.wind_dataset import pos_transform

class BaseGlobalEncoder(torch.nn.Module):
    """Base class for global encoders processing global data inputs."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config

    def forward(self, global_data: Dict[str, Any], local_data: Dict[str, Any]) -> torch.Tensor:
        """
        Args:
            global_data (dict): Dictionary of global data tensors.
            local_data (dict):
        Returns:
            torch.Tensor: Encoded global features at local positions (B, S, D).
        """
        raise NotImplementedError("Subclasses must implement the forward method.")
    
    
class SwinUNETRGlobalEncoder(BaseGlobalEncoder):
    """Global encoder using SwinUNETR architecture."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.model = SwinUNETR(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            spatial_dims=config["spatial_dims"],
        )
        self.field_extent = config['field_extent']

    def forward(self, global_data: Dict[str, Any], local_data: Dict[str, Any]) -> torch.Tensor:
        """
        Args:
            global_data (dict): Dictionary of global data tensors.
            local_data (dict):
        Returns:
            torch.Tensor: Encoded global features at local positions (B, S, D).
        """
        global_field = global_data["global_field"]  # Expecting shape (B, C, H, W, D)
        pad_typ, params = split_args("replicationpad")

        padding_x = global_field.shape[2] % 32
        padding_y = 32 - global_field.shape[3] % 32
        params['padding'] = (padding_y // 2, padding_y - padding_y // 2, padding_x // 2, padding_x - padding_x // 2)
        pad = Pad[pad_typ, 2](**params)
        global_field_padded = pad(global_field)
        output = self.model(global_field_padded)
        output = output[..., padding_x // 2: output.shape[-2] - (padding_x - padding_x // 2), padding_y // 2: output.shape[-1] - (padding_y - padding_y // 2)] 
        
        # interpolation
        pos_transformed = pos_transform(local_data['coords'], 
                                self.field_extent['min'], 
                                self.field_extent['max']).unsqueeze(2)
        emb = grid_sample(output, pos_transformed, align_corners=True, mode='bilinear', padding_mode='border')
        emb = emb.squeeze(-1).permute(0, 2, 1)
        return emb