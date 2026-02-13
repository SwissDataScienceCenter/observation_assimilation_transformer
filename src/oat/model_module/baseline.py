import torch
from torch import nn


from oat.model_module.global_encoder import InterpolationGlobalEncoder
from typing import Dict, Any

@dataclass
class InterpolaterBaselineConfig:
    field_extent: Dict[str, Any]
    mode: str = 'bilinear'
    padding_mode: str = 'border'
    

class InterpolaterBaseline(nn.Module):
    
    def __init__(self, config: InterpolaterBaselineConfig) -> None:
        super().__init__()
        self.register_buffer("field_min", torch.tensor(config.field_extent['min']))
        self.register_buffer("field_max", torch.tensor(config.field_extent['max']))
        interpolation_params = {
                'mode': config.mode,
                'padding_mode': config.padding_mode
            }
        self.interpolater = InterpolationGlobalEncoder(field_extent=config.field_extent, 
                                                       interpolation_params=interpolation_params)
        
    def forward(self, global_data: Dict[str, Any], 
        observation_data: Dict[str, Any],
        query_data: Dict[str, Any]) -> torch.Tensor:
        return self.interpolater(global_data, query_data)
    
    