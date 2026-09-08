import numpy as np
import torch

from oat.data_module.field import Field
from oat.predictor_module.oat_predictor import (
    _time_embedding_from_epoch_hours,
    _unwrap_fields_to_model_batch,
)

ROLES = {"target": "obs", "global": "nwp", "geo_bias": "geo", "topo": "topo"}


def _grp(n):
    coords = torch.zeros(2, n, 3)
    coords[..., 2] = 100.0  # epoch hours
    return {
        "fields": {
            "obs": Field(
                coords.clone(), torch.randn(2, n, 1), ["t"], torch.ones(2, n, 1)
            ),
            "nwp": Field(coords.clone(), torch.randn(2, n, 2), ["a", "b"]),
            "geo": Field(coords.clone(), torch.randn(2, n, 3), ["x", "y", "d"]),
            "topo": Field(coords.clone(), torch.randn(2, n, 5), list("pqrst")),
        }
    }


def test_time_embedding_parity():
    # 1970-01-01 05:00 UTC -> known day angle
    t = torch.tensor([5.0, 5.0])
    emb = _time_embedding_from_epoch_hours(t)
    assert emb.shape == (2, 4)
    ang = 2 * np.pi * (5 * 3600) / 86400
    assert abs(emb[0, 0].item() - np.sin(ang)) < 1e-5


def test_unwrap_shapes():
    batch = {"observation_data": _grp(3), "query_data": _grp(2)}
    out = _unwrap_fields_to_model_batch(batch, ROLES, requires_observation=True)
    od = out["observation_data"]
    assert od["coords"].shape == (2, 3, 2)  # x,y only
    assert od["topo"].shape == (2, 3, 5)  # no [.,C,1,1] reshape
    assert od["global_tokens"].shape == (2, 3, 2)
    assert od["obs"].shape == (2, 3, 1) and od["mask_obs"].shape == (2, 3, 1)
    assert out["query_data"]["geo"].shape == (2, 2, 3)
    assert out["global_data"]["time_emb"].shape == (2, 4)


def test_unwrap_uninformed_omits_obs():
    batch = {"observation_data": _grp(3), "query_data": _grp(2)}
    out = _unwrap_fields_to_model_batch(batch, ROLES, requires_observation=False)
    assert "observation_data" not in out


def _sphere_proj(coords):
    # Stand-in ERA5-style projection: (..., 2) -> (..., 3).
    lon = torch.deg2rad(coords[..., 0])
    lat = torch.deg2rad(coords[..., 1])
    cl = torch.cos(lat)
    return torch.stack(
        [cl * torch.cos(lon), cl * torch.sin(lon), torch.sin(lat)], dim=-1
    )


def test_unwrap_with_projection_emits_search_coords_and_projected_geo():
    from oat.predictor_module.oat_predictor import _unwrap_fields_to_model_batch

    batch = {"observation_data": _grp(3), "query_data": _grp(2)}
    out = _unwrap_fields_to_model_batch(
        batch, ROLES, requires_observation=True, project_index_coords=_sphere_proj
    )
    od = out["observation_data"]
    assert od["coords"].shape == (2, 3, 2)  # coords untouched (lon/lat)
    assert od["search_coords"].shape == (2, 3, 3)  # site A: projected xyz
    assert od["geo"].shape == (2, 3, 4)  # site B: [xyz, elevation]
    # first three geo channels equal the projected coords (unnormalized xyz)
    assert torch.allclose(od["geo"][..., :3], od["search_coords"], atol=1e-6)
    assert out["query_data"]["search_coords"].shape == (2, 2, 3)
    assert out["query_data"]["geo"].shape == (2, 2, 4)


def test_unwrap_without_projection_is_unchanged():
    from oat.predictor_module.oat_predictor import _unwrap_fields_to_model_batch

    batch = {"observation_data": _grp(3), "query_data": _grp(2)}
    base = _unwrap_fields_to_model_batch(batch, ROLES, requires_observation=True)
    proj_none = _unwrap_fields_to_model_batch(
        batch, ROLES, requires_observation=True, project_index_coords=None
    )
    od = proj_none["observation_data"]
    assert "search_coords" not in od
    assert od["geo"].shape == (2, 3, 3)
    assert torch.equal(od["geo"], base["observation_data"]["geo"])


def test_index_projection_dim():
    from oat.predictor_module.field_batch_utils import index_projection_dim

    assert index_projection_dim(None) == 2
    assert index_projection_dim(_sphere_proj) == 3


def test_resolve_index_projection_override_vs_default():
    from oat.data_module.dataset import DatasetBase
    from oat.predictor_module.field_batch_utils import resolve_index_projection

    class _Plain(DatasetBase):
        def __init__(self):  # bypass DatasetBase.__init__ machinery
            pass

    class _Geo(DatasetBase):
        def __init__(self):
            pass

        def project_index_coords(self, coords):
            return _sphere_proj(coords)

    assert resolve_index_projection(_Plain()) is None
    geo = _Geo()
    assert resolve_index_projection(geo) is not None

    # unwrap one level: a wrapper exposing `.dataset`
    class _Wrap:
        def __init__(self, inner):
            self.dataset = inner

    assert resolve_index_projection(_Wrap(geo)) is not None
    assert resolve_index_projection(_Wrap(_Plain())) is None
