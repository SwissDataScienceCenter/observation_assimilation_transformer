import pytest
import torch

from oat.data_module.field import Field


def _unbatched():
    return Field(
        coords=torch.zeros(5, 3),
        values=torch.zeros(5, 2),
        var_names=["a", "b"],
    )


def _batched():
    return Field(
        coords=torch.zeros(4, 5, 3),
        values=torch.zeros(4, 5, 2),
        var_names=["a", "b"],
        mask=torch.ones(4, 5, 2),
    )


def test_unbatched_properties():
    f = _unbatched()
    assert f.num_points == 5
    assert f.num_channels == 2
    assert f.spatial_coords.shape == (5, 2)
    torch.testing.assert_close(f.spatial_coords, f.coords[:, :2])
    assert f.mask is None


def test_batched_properties():
    f = _batched()
    assert f.num_points == 5
    assert f.num_channels == 2
    assert f.spatial_coords.shape == (4, 5, 2)
    torch.testing.assert_close(f.spatial_coords, f.coords[..., :2])


def test_var_names_length_must_match_channels():
    with pytest.raises(ValueError, match="var_names"):
        Field(coords=torch.zeros(5, 3), values=torch.zeros(5, 2), var_names=["a"])


def test_coord_dim_must_be_three():
    with pytest.raises(ValueError, match="coords"):
        Field(coords=torch.zeros(5, 2), values=torch.zeros(5, 2), var_names=["a", "b"])


def test_point_count_must_match():
    with pytest.raises(ValueError, match="points"):
        Field(coords=torch.zeros(6, 3), values=torch.zeros(5, 2), var_names=["a", "b"])


def test_batch_dim_must_match():
    with pytest.raises(ValueError, match="batch"):
        Field(
            coords=torch.zeros(3, 5, 3),
            values=torch.zeros(4, 5, 2),
            var_names=["a", "b"],
        )


def test_mask_shape_must_match_values():
    with pytest.raises(ValueError, match="mask"):
        Field(
            coords=torch.zeros(5, 3),
            values=torch.zeros(5, 2),
            var_names=["a", "b"],
            mask=torch.ones(5, 1),
        )


def test_mask_none_is_preserved():
    f = _unbatched()
    assert f.mask is None
