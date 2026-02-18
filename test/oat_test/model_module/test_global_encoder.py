import os
import pytest
import torch
from oat.model_module.global_encoder import (
    SwinUNETRGlobalEncoder, UNetGlobalEncoder, BaseGlobalEncoder,
    MLPGlobalEncoder, InterpolationGlobalEncoder,
)
from typing import Any, Dict


def swin_unetr_params() -> Dict[str, Any]:
    return {
        'out_channels': 4,
        'spatial_dims': 2,
        'patch_size': 2,
    }
    
    
def unet_params() -> Dict[str, Any]:
    return {
        'out_channels': 4,
        'spatial_dims': 2,
        'channels': (8, 16, 32, 64),
        'strides': (1, 1, 1),
        'num_res_units': 2,
    }
    

def mlp_params() -> Dict[str, Any]:
    return {
        'in_channels': 4,
        'hidden_dims': [16, 32],
        'out_channels': 4,
        'act_fn': 'ReLU'
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


@pytest.fixture(params=["swin_unetr", "unet", "mlp"])
def encoder(request: pytest.FixtureRequest, img: torch.Tensor, field_extent: Dict[str, torch.Tensor]) -> BaseGlobalEncoder:
    if request.param == "swin_unetr":
        params = swin_unetr_params()
        params['in_channels'] = img.shape[-3]
        return SwinUNETRGlobalEncoder(
            swin_unetr_params=params,
            field_extent=field_extent
        )
    elif request.param == "unet":
        params = unet_params()
        params['in_channels'] = img.shape[-3]
        return UNetGlobalEncoder(
            unet_params=params,
            field_extent=field_extent
        )
    elif request.param == "mlp":
        params = mlp_params()
        params['in_channels'] = img.shape[-3]
        return MLPGlobalEncoder(
            mlp_params=params,
            field_extent=field_extent
        )

@pytest.fixture(params=["swin_unetr", "unet", "mlp"])
def encoder_from_batch(request: pytest.FixtureRequest, batch_fixture: dict[str, Any], field_extent: Dict[str, torch.Tensor]) -> BaseGlobalEncoder:
    if request.param == "swin_unetr":
        params = swin_unetr_params()
        params['in_channels'] = batch_fixture["global_data"]["global_field"].shape[-3]
        return SwinUNETRGlobalEncoder(
            swin_unetr_params=params,
            field_extent=field_extent
        )
    elif request.param == "unet":
        params = unet_params()
        params['in_channels'] = batch_fixture["global_data"]["global_field"].shape[-3]
        return UNetGlobalEncoder(
            unet_params=params,
            field_extent=field_extent
        )
    elif request.param == "mlp":
        params = mlp_params()
        params['in_channels'] = batch_fixture["global_data"]["global_field"].shape[-3]
        return MLPGlobalEncoder(
            mlp_params=params,
            field_extent=field_extent
        )


def test_global_encoder_initialization(encoder: object, field_extent: Dict[str, torch.Tensor]) -> None:
    assert encoder.model is not None

    assert (encoder.field_min == field_extent['min']).all()
    assert (encoder.field_max == field_extent['max']).all()


def test_pad_forward(encoder: BaseGlobalEncoder, img: torch.Tensor) -> None:
    if not hasattr(encoder, "_pad"):
        pytest.skip("Encoder does not implement _pad method")
    for padding_x, padding_y in [(0, 0), (2, 2), (3, 5), (4, 4), (7, 1)]:
        padded = encoder._pad(img, padding_x, padding_y, unpad=False)
        assert padded.shape[-2] == img.shape[-2] + padding_x
        assert padded.shape[-1] == img.shape[-1] + padding_y

def test_pad_unpad_roundtrip(encoder: BaseGlobalEncoder, img: torch.Tensor) -> None:
    
    if not hasattr(encoder, "_pad"):
        pytest.skip("Encoder does not implement _pad method")
    for padding_x, padding_y in [(0, 0), (2, 2), (3, 5), (4, 4), (7, 1)]:
        padded = encoder._pad(img, padding_x, padding_y, unpad=False)
        unpadded = encoder._pad(padded, padding_x, padding_y, unpad=True)
        
        assert unpadded.shape == img.shape

def test_forward_output_shape(encoder: BaseGlobalEncoder, img: torch.Tensor) -> None:
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


def test_forward_different_batch_sizes(encoder: BaseGlobalEncoder, img: torch.Tensor) -> None:
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


def test_forward_different_station_counts(encoder: BaseGlobalEncoder, img: torch.Tensor) -> None:
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
        
def test_unet_output_size(encoder: BaseGlobalEncoder, img: torch.Tensor) -> None:
    if not isinstance(encoder, UNetGlobalEncoder):
        pytest.skip("Test only applicable to UNetGlobalEncoder")
    batch_size = 2
    global_data = {
        "global_field": img[:batch_size]
    }
    output = encoder.apply_model(global_data)
    expected_height = img.shape[-2]
    expected_width = img.shape[-1]
    assert output.shape[-2] == expected_height
    assert output.shape[-1] == expected_width
        

@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Too heavy for CI"
)
def test_with_batch(encoder_from_batch: BaseGlobalEncoder, batch_fixture: dict[str, Any]) -> None:
    global_data = batch_fixture["global_data"]
    local_data = batch_fixture["observation_data"]
    batch_size, num_stations = local_data["coords"].shape[:2]
    output = encoder_from_batch(global_data, local_data)
    assert output.shape == (batch_size, num_stations, 4)


def test_interpolation_encoder_output_shape(field_extent: Dict[str, torch.Tensor]) -> None:
    B, C, H, W, S = 2, 5, 32, 32, 10
    encoder = InterpolationGlobalEncoder(field_extent=field_extent)
    global_data = {"global_field": torch.randn(B, C, H, W)}
    local_data = {"coords": torch.rand(B, S, 2) * 100.0}
    out = encoder(global_data, local_data)
    assert out.shape == (B, S, C)


def test_interpolation_encoder_varies_batch_size(field_extent: Dict[str, torch.Tensor]) -> None:
    C, H, W, S = 4, 16, 16, 8
    encoder = InterpolationGlobalEncoder(field_extent=field_extent)
    for B in (1, 2, 4):
        out = encoder(
            {"global_field": torch.randn(B, C, H, W)},
            {"coords": torch.rand(B, S, 2) * 100.0},
        )
        assert out.shape[0] == B


def test_interpolation_encoder_varies_station_count(field_extent: Dict[str, torch.Tensor]) -> None:
    B, C, H, W = 2, 3, 16, 16
    encoder = InterpolationGlobalEncoder(field_extent=field_extent)
    for S in (1, 10, 50):
        out = encoder(
            {"global_field": torch.randn(B, C, H, W)},
            {"coords": torch.rand(B, S, 2) * 100.0},
        )
        assert out.shape[1] == S


@pytest.mark.skipif(os.getenv("CI") == "true", reason="Too heavy for CI")
def test_interpolation_encoder_with_batch_fixture(batch_fixture: dict[str, Any], field_extent: Dict[str, torch.Tensor]) -> None:
    global_data = batch_fixture["global_data"]
    query_data = batch_fixture["query_data"]
    B = global_data["global_field"].shape[0]
    C_global = global_data["global_field"].shape[1]
    S = query_data["coords"].shape[1]
    encoder = InterpolationGlobalEncoder(field_extent=field_extent)
    out = encoder(global_data, query_data)
    assert out.shape == (B, S, C_global)