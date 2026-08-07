import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from oat.data_module.field import Field
from oat.model_module.field_sampler import build_index, nearest_read


def test_build_index_returns_ckdtree_over_xy():
    coords = torch.tensor(
        [[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0]]
    )
    tree = build_index(coords)
    assert isinstance(tree, cKDTree)
    # tree indexes only (x, y): querying (0.9, 0.0) returns point 1
    _, idx = tree.query(np.array([[0.9, 0.0]]))
    assert idx[0] == 1


def test_build_index_uses_batch_element_zero():
    coords = torch.zeros(4, 3, 3)
    coords[0] = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    tree = build_index(coords)
    assert tree.n == 3


def test_build_index_rejects_unknown_distance():
    with pytest.raises(NotImplementedError, match="euclidean"):
        build_index(torch.zeros(3, 3), distance="haversine")


def _square_field():
    # 4 corners of the unit square, distinct per-corner values, constant t=5
    coords = torch.tensor(
        [[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0], [1.0, 1.0, 5.0]]
    )
    values = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    return Field(coords=coords, values=values, var_names=["v"])


def test_nearest_read_unbatched_picks_corners():
    f = _square_field()
    dst = torch.tensor([[0.1, 0.1, 5.0], [0.9, 0.9, 5.0], [0.05, 0.9, 5.0]])
    out = nearest_read(f, dst)
    assert out.shape == (3, 1)
    torch.testing.assert_close(out, torch.tensor([[10.0], [40.0], [30.0]]))


def test_nearest_read_ignores_time_column():
    f = _square_field()
    # same (x, y) as above but a wildly different t must not change selection
    dst = torch.tensor([[0.1, 0.1, -999.0], [0.9, 0.9, 1e6, ]])
    out = nearest_read(f, dst)
    torch.testing.assert_close(out, torch.tensor([[10.0], [40.0]]))


def test_nearest_read_batched():
    f = _square_field()
    coords_b = f.coords.unsqueeze(0).repeat(2, 1, 1)
    values_b = torch.stack([f.values, f.values + 100.0], dim=0)  # [2, 4, 1]
    fb = Field(coords=coords_b, values=values_b, var_names=["v"])
    dst = torch.tensor(
        [
            [[0.1, 0.1, 5.0], [0.9, 0.9, 5.0]],  # -> 10, 40
            [[0.9, 0.1, 5.0], [0.1, 0.9, 5.0]],  # -> 120, 130
        ]
    )
    out = nearest_read(fb, dst)
    assert out.shape == (2, 2, 1)
    torch.testing.assert_close(
        out, torch.tensor([[[10.0], [40.0]], [[120.0], [130.0]]])
    )


def test_nearest_read_prebuilt_index_matches():
    f = _square_field()
    dst = torch.tensor([[0.1, 0.1, 5.0], [0.9, 0.9, 5.0]])
    idx = build_index(f.coords)
    torch.testing.assert_close(nearest_read(f, dst, index=idx), nearest_read(f, dst))


def test_nearest_read_single_source_point():
    f = Field(
        coords=torch.tensor([[3.0, 3.0, 0.0]]),
        values=torch.tensor([[7.0]]),
        var_names=["v"],
    )
    dst = torch.tensor([[0.0, 0.0, 0.0], [100.0, 100.0, 0.0]])
    out = nearest_read(f, dst)
    torch.testing.assert_close(out, torch.tensor([[7.0], [7.0]]))


def test_nearest_read_identity_when_dst_equals_source():
    f = _square_field()
    out = nearest_read(f, f.coords)
    torch.testing.assert_close(out, f.values)


def test_nearest_read_output_on_values_device():
    f = _square_field()
    dst = torch.tensor([[0.1, 0.1, 5.0]])
    out = nearest_read(f, dst)
    assert out.device == f.values.device


def test_nearest_read_rejects_unknown_distance():
    f = _square_field()
    with pytest.raises(NotImplementedError, match="euclidean"):
        nearest_read(f, f.coords, distance="haversine")
