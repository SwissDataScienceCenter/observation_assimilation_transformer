import torch
from torch import nn


from oat.model_module.global_encoder import InterpolationGlobalEncoder
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class InterpolatorBaselineConfig:
    field_extent: Dict[str, Any]
    mode: str = 'bilinear'
    padding_mode: str = 'border'
    

class InterpolatorBaseline(nn.Module):
    
    def __init__(self, config: InterpolatorBaselineConfig) -> None:
        super().__init__()
        interpolation_params = {
                'mode': config.mode,
                'padding_mode': config.padding_mode
            }
        self.interpolator = InterpolationGlobalEncoder(field_extent=config.field_extent, 
                                                       interpolation_params=interpolation_params)
        
    def forward(self, global_data: Dict[str, Any], 
        observation_data: Dict[str, Any],
        query_data: Dict[str, Any]) -> torch.Tensor:
        return self.interpolator(global_data, query_data)
    
    