import pytest
import torch

from oat.model_module.global_encoder import (
    InterpolationGlobalEncoder,
    MLPGlobalEncoder,
    ResNetCNNGlobalEncoder,
    SwinUNETRGlobalEncoder,
    UNetGlobalEncoder,
)


@pytest.fixture
def field_extent() -> dict[str, torch.Tensor]:
    return {
        "min": torch.tensor([0.0, 0.0]),
        "max": torch.tensor([100.0, 100.0]),
    }


def test_interpolation_global_encoder_is_deprecated(
    field_extent: dict[str, torch.Tensor],
) -> None:
    with pytest.raises(NotImplementedError, match="deprecated"):
        InterpolationGlobalEncoder(field_extent=field_extent)


def test_mlp_global_encoder_is_deprecated(
    field_extent: dict[str, torch.Tensor],
) -> None:
    with pytest.raises(NotImplementedError, match="deprecated"):
        MLPGlobalEncoder(
            mlp_params={"in_channels": 4, "out_channels": 4},
            field_extent=field_extent,
        )


def test_unet_global_encoder_is_deprecated(
    field_extent: dict[str, torch.Tensor],
) -> None:
    with pytest.raises(NotImplementedError, match="deprecated"):
        UNetGlobalEncoder(unet_params={}, field_extent=field_extent)


def test_resnet_cnn_global_encoder_is_deprecated(
    field_extent: dict[str, torch.Tensor],
) -> None:
    with pytest.raises(NotImplementedError, match="deprecated"):
        ResNetCNNGlobalEncoder(resnet_cnn_params={}, field_extent=field_extent)


def test_swin_unetr_global_encoder_is_deprecated(
    field_extent: dict[str, torch.Tensor],
) -> None:
    with pytest.raises(NotImplementedError, match="deprecated"):
        SwinUNETRGlobalEncoder(swin_unetr_params={}, field_extent=field_extent)
