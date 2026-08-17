import torch

from oat.data_module.dataset import DatasetBase


def _sphere_proj(coords):
    lon = torch.deg2rad(coords[..., 0])
    lat = torch.deg2rad(coords[..., 1])
    cl = torch.cos(lat)
    return torch.stack(
        [cl * torch.cos(lon), cl * torch.sin(lon), torch.sin(lat)], dim=-1
    )


def test_d_geo_bias_derivation_identity_and_sphere():
    from oat.predictor_module.field_batch_utils import index_projection_dim

    geo_vars = ["longitude", "latitude", "elevation"]  # len 3
    # identity projection -> proj_dim 2 -> d_geo_bias == len(geo_vars)
    d_identity = len(geo_vars) - 2 + index_projection_dim(None)
    assert d_identity == 3
    # sphere projection -> proj_dim 3 -> d_geo_bias == len(geo_vars) + 1
    d_sphere = len(geo_vars) - 2 + index_projection_dim(_sphere_proj)
    assert d_sphere == 4


def test_set_index_projection_stores_override():
    from oat.predictor_module.base_predictor import BasePredictor

    class _Stub(BasePredictor):
        def __init__(self):
            super().__init__(output_var_names=["t"], field_roles={"a": "b"})

        def _normalize_batch(self, batch):
            return batch

    class _GeoDataset(DatasetBase):
        def __init__(self):
            pass

        def project_index_coords(self, coords):
            return _sphere_proj(coords)

    class _PlainDataset(DatasetBase):
        def __init__(self):
            pass

    p = _Stub()
    assert p.project_index_coords is None  # default
    p.set_index_projection(_PlainDataset())
    assert p.project_index_coords is None  # non-overriding -> stays None
    p.set_index_projection(_GeoDataset())
    assert p.project_index_coords is not None
    out = p.project_index_coords(torch.tensor([[0.0, 90.0]]))
    assert torch.allclose(out[0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)
