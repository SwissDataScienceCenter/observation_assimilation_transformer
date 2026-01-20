import pytest
import torch
from oat.model_module.global_encoder import SwinUNETRGlobalEncoder
from typing import Any, Dict

@pytest.fixture
def swin_unetr_params() -> Dict[str, Any]:
    return {
        'out_channels': 32,
        'spatial_dims': 2,
        'patch_size': 2,
    }


@pytest.fixture
def field_extent() -> Dict[str, torch.Tensor]:
    return {
        "min": torch.tensor([0.0, 0.0]),
        "max": torch.tensor([100.0, 100.0]),
    }

@pytest.fixture
def img() -> torch.Tensor:
    return torch.randn(16, 20, 800, 904)

@pytest.fixture
def encoder(img: torch.Tensor, swin_unetr_params: Dict[str, Any], field_extent: Dict[str, torch.Tensor]) -> SwinUNETRGlobalEncoder:
    swin_unetr_params['in_channels'] = img.shape[-3]
    return SwinUNETRGlobalEncoder(
        swin_unetr_params=swin_unetr_params,
        field_extent=field_extent
    )


def test_swin_unetr_global_encoder_initialization(encoder: SwinUNETRGlobalEncoder, 
                                                  field_extent: Dict[str, torch.Tensor]) -> None:
    assert encoder.model is not None
    assert encoder.field_extent == field_extent


def test_pad_forward(encoder: SwinUNETRGlobalEncoder, img: torch.Tensor) -> None:
    padding_x, padding_y = 4, 4
    
    padded = encoder._pad(img, padding_x, padding_y, unpad=False)
    assert padded.shape[-2] == img.shape[-2] + padding_x
    assert padded.shape[-1] == img.shape[-1] + padding_y


def test_pad_unpad_roundtrip(encoder: SwinUNETRGlobalEncoder, img: torch.Tensor) -> None:
    padding_x, padding_y = 4, 4
    
    padded = encoder._pad(img, padding_x, padding_y, unpad=False)
    unpadded = encoder._pad(padded, padding_x, padding_y, unpad=True)
    
    assert unpadded.shape == img.shape


def test_forward_output_shape(encoder: SwinUNETRGlobalEncoder, img: torch.Tensor) -> None:
    batch_size = 2
    num_stations = 10
    global_data = {
        "global_field": img[:batch_size]
    }
    local_data = {
        "coords": torch.rand(batch_size, num_stations, 2) * 100.0
    }
    
    output = encoder(global_data, local_data)
    
    assert output.shape[0] == batch_size
    assert output.shape[1] == num_stations
    assert output.shape[2] == 32  # out_channels from swin_unetr_params


def test_forward_different_batch_sizes(encoder: SwinUNETRGlobalEncoder, img: torch.Tensor) -> None:
    for batch_size in [1, 4, 8]:
        num_stations = 5
        global_data = {
            "global_field": img[:batch_size]
        }
        local_data = {
            "coords": torch.rand(batch_size, num_stations, 2) * 100.0
        }
        
        output = encoder(global_data, local_data)
        assert output.shape[0] == batch_size


def test_forward_different_station_counts(encoder: SwinUNETRGlobalEncoder, img: torch.Tensor) -> None:
    batch_size = 2
    for num_stations in [1, 10, 50]:
        global_data = {
            "global_field": img[:batch_size]
        }
        local_data = {
            "coords": torch.rand(batch_size, num_stations, 2) * 100.0
        }
        
        output = encoder(global_data, local_data)
        assert output.shape[1] == num_stations