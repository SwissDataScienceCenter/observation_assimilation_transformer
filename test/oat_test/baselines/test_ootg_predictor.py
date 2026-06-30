import sys
from pathlib import Path

import numpy as np
import torch

_TNPDIR = Path(__file__).parents[4] / "deps" / "gridded-tnp"
if _TNPDIR.exists() and str(_TNPDIR) not in sys.path:
    sys.path.insert(0, str(_TNPDIR))

from oat.baselines.coord_utils import lv95_to_latlon, nwp_latlon_grid
from oat.baselines.ootg_model import build_ootg_model
from oat.baselines.ootg_predictor import OOTGPredictor


def _make_tiny_predictor(dim_y=1, dim_gridded_y=2, v2d=None):
    if v2d is None:
        v2d = {"temperature": "normal"}
    model = build_ootg_model(
        dim_y=dim_y, dim_gridded_y=dim_gridded_y, obs_variable_to_distribution=v2d,
        embed_dim=32, x_embed_dim=32, num_heads=2, head_dim=8,
        num_layers=2, num_legendre_polys=4, grid_shape=(8, 8),
        window_sizes=(4, 4), shift_sizes=(2, 2), top_k_ctot=4,
    )
    ref_x = np.linspace(2255500.0, 2964500.0, 80)
    ref_y = np.linspace(840500.0, 1479500.0, 80)
    xc_grid = nwp_latlon_grid(ref_x, ref_y, stride=10)   # (8, 8, 2)
    obs_vars = list(v2d.keys())
    dist_names = list(v2d.values())
    return OOTGPredictor(
        model=model, obs_vars=obs_vars,
        obs_mean=torch.full((dim_y,), 280.0),
        obs_std=torch.full((dim_y,), 5.0),
        nwp_mean=torch.zeros(dim_gridded_y),
        nwp_std=torch.ones(dim_gridded_y),
        xc_grid_base=xc_grid,
        coord_to_latlon=lv95_to_latlon,
        dist_names=dist_names,
        n_nwp=dim_gridded_y,
        nwp_stride=10,
    )


def _make_oat_batch(B=1, N_obs=15, N_q=5, dim_y=1, n_nwp=2):
    coords = torch.rand(B, N_obs, 2) * 1e5 + torch.tensor([2.5e6, 1.1e6])
    q_coords = torch.rand(B, N_q, 2) * 1e5 + torch.tensor([2.5e6, 1.1e6])
    obs = torch.zeros(B, N_obs, dim_y)
    obs[..., 0] = torch.randn(B, N_obs) * 5.0 + 280.0
    if dim_y == 2:
        obs[..., 1] = torch.rand(B, N_obs) * 5.0
    q_obs = obs.clone()[:, :N_q]
    mask = torch.ones(B, N_obs, dim_y)
    mask[:, 0] = 0.0
    q_mask = torch.ones(B, N_q, dim_y)
    return {
        "observation_data": {"obs": obs, "mask_obs": mask, "coords": coords},
        "query_data": {"obs": q_obs, "mask_obs": q_mask, "coords": q_coords},
        "global_data": {"global_field": torch.randn(B, n_nwp + 1, 80, 80)},
    }


def test_predict_step_shape_univariate():
    pred = _make_tiny_predictor(dim_y=1)
    out = pred.predict_step(_make_oat_batch(B=1, N_q=5, dim_y=1), 0)
    assert out.shape == (1, 5, 1, 1)


def test_predict_step_shape_mixed():
    v2d = {"temperature": "normal", "precipitation": "zero_inflated_lognormal"}
    pred = _make_tiny_predictor(dim_y=2, dim_gridded_y=4, v2d=v2d)
    out = pred.predict_step(_make_oat_batch(B=1, N_q=5, dim_y=2, n_nwp=4), 0)
    assert out.shape == (1, 5, 1, 2)


def test_predict_step_n_samples():
    pred = _make_tiny_predictor(dim_y=1)
    out = pred.predict_step(_make_oat_batch(B=1, N_q=5), 0, n_samples=10)
    assert out.shape == (1, 5, 10, 1)


def test_temperature_in_original_units():
    pred = _make_tiny_predictor(dim_y=1)
    out = pred.predict_step(_make_oat_batch(B=1, N_q=5), 0)
    # temperature should be ~280 K, not normalized ~0
    assert out[..., 0].abs().mean() > 10.0


def test_precipitation_non_negative():
    v2d = {"temperature": "normal", "precipitation": "zero_inflated_lognormal"}
    pred = _make_tiny_predictor(dim_y=2, dim_gridded_y=4, v2d=v2d)
    # sample many times to check ZILN samples are non-negative
    out = pred.predict_step(_make_oat_batch(B=1, N_q=5, dim_y=2, n_nwp=4), 0, n_samples=20)
    assert (out[..., 1] >= 0).all(), "ZILN samples must be non-negative"


def test_normalize_batch_is_identity():
    pred = _make_tiny_predictor()
    batch = _make_oat_batch()
    assert pred._normalize_batch(batch) is batch


def test_output_var_names():
    v2d = {"temperature": "normal", "precipitation": "zero_inflated_lognormal"}
    pred = _make_tiny_predictor(dim_y=2, v2d=v2d)
    assert pred.output_var_names == ["temperature", "precipitation"]
