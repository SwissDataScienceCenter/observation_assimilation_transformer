import os
import pytest
import torch
from oat.model_module.global_encoder import SwinUNETRGlobalEncoder, UNetGlobalEncoder
from typing import Any, Dict

@pytest.fixture
def swin_unetr_params() -> Dict[str, Any]:
    return {
        'out_channels': 4,
        'spatial_dims': 2,
        'patch_size': 2,
    }
    
    
@pytest.fixture
def unet_params() -> Dict[str, Any]:
    return {
        'out_channels': 4,
        'spatial_dims': 2,
        'channels': (8, 16, 32, 64),
        'strides': (1, 1, 1),
        'num_res_units': 2,
    }


@pytest.fixture
def field_extent() -> Dict[str, torch.Tensor]:
    return {
        "min": torch.tensor([0.0, 0.0]),
        "max": torch.tensor([100.0, 100.0]),
    }

@pytest.fixture(params=[
    (4, 10, 200, 129),
    (4, 5, 128, 256),
])
def img(request: pytest.FixtureRequest) -> torch.Tensor:
    return torch.randn(*request.param)


@pytest.fixture
def encoder(img: torch.Tensor, swin_unetr_params: Dict[str, Any], field_extent: Dict[str, torch.Tensor]) -> SwinUNETRGlobalEncoder:
    params = swin_unetr_params.copy()
    params['in_channels'] = img.shape[-3]
    return SwinUNETRGlobalEncoder(
        swin_unetr_params=params,
        field_extent=field_extent
    )

@pytest.fixture
def encoder_from_batch(batch_fixture: dict[str, Any], swin_unetr_params: Dict[str, Any], field_extent: Dict[str, torch.Tensor]) -> SwinUNETRGlobalEncoder:
    params = swin_unetr_params.copy()
    params['in_channels'] = batch_fixture["global_data"]["global_field"].shape[-3]
    return SwinUNETRGlobalEncoder(
        swin_unetr_params=params,
        field_extent=field_extent
    )
    
@pytest.fixture
def unet_encoder(img: torch.Tensor, unet_params: Dict[str, Any], field_extent: Dict[str, torch.Tensor]) -> UNetGlobalEncoder:
    params = unet_params.copy()
    params['in_channels'] = img.shape[-3]
    return UNetGlobalEncoder(
        unet_params=params,
        field_extent=field_extent
    )
    
@pytest.fixture
def unet_encoder_from_batch(batch_fixture: dict[str, Any], unet_params: Dict[str, Any], field_extent: Dict[str, torch.Tensor]) -> UNetGlobalEncoder:
    params = unet_params.copy()
    params['in_channels'] = batch_fixture["global_data"]["global_field"].shape[-3]
    return UNetGlobalEncoder(
        unet_params=params,
        field_extent=field_extent
    )

@pytest.mark.parametrize("encoder", ["encoder", "unet_encoder"], indirect=True)
def test_swin_unetr_global_encoder_initialization(encoder: object, field_extent: Dict[str, torch.Tensor]) -> None:
    assert encoder.model is not None
    assert encoder.field_extent == field_extent


def test_pad_forward(encoder: SwinUNETRGlobalEncoder, img: torch.Tensor) -> None:

    for padding_x, padding_y in [(0, 0), (2, 2), (3, 5), (4, 4), (7, 1)]:
        padded = encoder._pad(img, padding_x, padding_y, unpad=False)
        assert padded.shape[-2] == img.shape[-2] + padding_x
        assert padded.shape[-1] == img.shape[-1] + padding_y

def test_pad_unpad_roundtrip(encoder: SwinUNETRGlobalEncoder, img: torch.Tensor) -> None:
    
    for padding_x, padding_y in [(0, 0), (2, 2), (3, 5), (4, 4), (7, 1)]:
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
    assert output.shape[2] == 4  # out_channels from swin_unetr_params


@pytest.mark.parametrize("encoder", ["encoder", "unet_encoder"], indirect=True)
def test_forward_different_batch_sizes(encoder: object, img: torch.Tensor) -> None:
    for batch_size in [1, 3, 4,]:
        num_stations = 5
        global_data = {
            "global_field": img[:batch_size]
        }
        local_data = {
            "coords": torch.rand(batch_size, num_stations, 2) * 100.0
        }
        
        output = encoder(global_data, local_data)
        assert output.shape[0] == batch_size


@pytest.mark.parametrize("encoder", ["encoder", "unet_encoder"], indirect=True)
def test_forward_different_station_counts(encoder: object, img: torch.Tensor) -> None:
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
        

@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Too heavy for CI"
)        
@pytest.mark.parametrize("encoder_from_batch", ["encoder_from_batch", "unet_encoder_from_batch"], indirect=True)
def test_with_batch(encoder_from_batch: object, batch_fixture: dict[str, Any]) -> None:
    global_data = batch_fixture["global_data"]
    local_data = batch_fixture["observation_data"]
    batch_size, num_stations = local_data["coords"].shape[:2]
    output = encoder_from_batch(global_data, local_data)
    assert output.shape == (batch_size, num_stations, 4)