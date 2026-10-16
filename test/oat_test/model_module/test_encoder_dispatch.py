"""Encoder dispatch accepts only the live encoder types."""

import pytest

from oat.model_module import model as M


def test_global_encoder_config_defaults_to_point():
    assert M.GlobalEncoderConfig(in_channels=2).type == "Point"


def test_global_encoder_config_has_no_field_extent():
    cfg = M.GlobalEncoderConfig(in_channels=2)
    assert not hasattr(cfg, "field_extent_min")
    assert not hasattr(cfg, "field_extent_max")


def test_topographic_encoder_config_defaults_to_mlp():
    assert M.TopographicEncoderConfig(in_channels=5).type == "MLP"


@pytest.mark.parametrize("bad_type", ["UNet", "SwinUNETR", "MLP", "ResNetCNN"])
def test_removed_global_encoder_types_raise_value_error(
    bad_type, uninformed_model_cfg_fixture
):
    uninformed_model_cfg_fixture.global_encoder.type = bad_type
    with pytest.raises(ValueError, match="Unsupported global encoder type"):
        M.UninformedOAT(uninformed_model_cfg_fixture)


def test_removed_topographic_encoder_type_raises_value_error():
    tc = M.TopographicEncoderConfig(in_channels=5, type="CNN")
    with pytest.raises(ValueError, match="Unsupported topographic encoder type"):
        M.build_topographic_encoder(tc)


def test_cnn_topography_encoder_is_gone():
    from oat.model_module import topographic_encoder

    assert not hasattr(topographic_encoder, "CNNTopographyEncoder")
