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
