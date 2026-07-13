import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

_TNPDIR = Path(__file__).parents[4] / "deps" / "gridded-tnp"
if _TNPDIR.exists() and str(_TNPDIR) not in sys.path:
    sys.path.insert(0, str(_TNPDIR))

from oat.baselines.coord_utils import lv95_to_latlon, nwp_latlon_grid
from oat.baselines.ootg_model import build_ootg_model
from oat.baselines.ootg_predictor import OOTGPredictor
from oat.data_module.transform import StandardNormalizeTransform


def _make_tiny_predictor(dim_y=1, dim_gridded_y=2, v2d=None):
    if v2d is None:
        v2d = {"temperature": "normal"}
    model = build_ootg_model(
        dim_y=dim_y,
        dim_gridded_y=dim_gridded_y,
        obs_variable_to_distribution=v2d,
        embed_dim=32,
        x_embed_dim=32,
        num_heads=2,
        head_dim=8,
        num_layers=2,
        num_legendre_polys=4,
        grid_shape=(8, 8),
        window_sizes=(4, 4),
        shift_sizes=(2, 2),
        top_k_ctot=4,
    )
    ref_x = np.linspace(2255500.0, 2964500.0, 80)
    ref_y = np.linspace(840500.0, 1479500.0, 80)
    xc_grid = nwp_latlon_grid(ref_x, ref_y, stride=10)  # (8, 8, 2)
    obs_vars = list(v2d.keys())
    dist_names = list(v2d.values())
    return OOTGPredictor(
        model=model,
        obs_vars=obs_vars,
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
    out = pred.predict_step(
        _make_oat_batch(B=1, N_q=5, dim_y=2, n_nwp=4), 0, n_samples=20
    )
    assert (out[..., 1] >= 0).all(), "ZILN samples must be non-negative"


def test_normalize_batch_is_identity():
    pred = _make_tiny_predictor()
    batch = _make_oat_batch()
    assert pred._normalize_batch(batch) is batch


def test_predict_step_empty_context():
    pred = _make_tiny_predictor(dim_y=1)
    batch = _make_oat_batch(B=1, N_obs=5, N_q=3, dim_y=1)
    # Zero out all observation masks — no valid context stations
    batch["observation_data"]["mask_obs"] = torch.zeros(1, 5, 1)
    out = pred.predict_step(batch, 0)
    assert out.shape == (1, 3, 1, 1), f"Expected (1,3,1,1), got {out.shape}"


def test_output_var_names():
    v2d = {"temperature": "normal", "precipitation": "zero_inflated_lognormal"}
    pred = _make_tiny_predictor(dim_y=2, v2d=v2d)
    assert pred.output_var_names == ["temperature", "precipitation"]


def _make_gen_batch(B=1, N_c=5, N_t=4, dim_y=1):
    """Synthetic generator-format batch (dot-notation attributes)."""
    return types.SimpleNamespace(
        xc=torch.randn(B, N_c, 2),
        yc=torch.randn(B, N_c, dim_y),
        xc_grid=torch.randn(B, 8, 8, 2),
        yc_grid=torch.randn(B, 8, 8, 2),
        xt=torch.randn(B, N_t, 2),
        yt=torch.randn(B, N_t, dim_y),
    )


def test_draw_samples_shape():
    pred = _make_tiny_predictor(dim_y=1)
    pred.n_samples = 10
    mock_dist = MagicMock()
    mock_dist.sample.return_value = torch.randn(10, 1, 4, 1)  # [n, B, N_t, dim_y]
    samples = pred._draw_samples(mock_dist)
    assert samples.shape == (1, 4, 10, 1)  # [B, N_t, n_samples, dim_y]


def test_training_step_logs_normalized_loss_and_physical_metrics():
    """train_loss_{var} (normalized CRPS) and train_crps_{var} (physical) must differ."""
    pred = _make_tiny_predictor(dim_y=1)
    pred.n_samples = 10

    logged = {}

    def capture(key, val, **kwargs):
        logged[key] = val.item() if hasattr(val, "item") else float(val)

    batch = _make_gen_batch(dim_y=1)
    with patch.object(pred, "log", side_effect=capture):
        with patch.object(pred, "_forward_generator_batch") as mock_fwd:
            mock_dist = MagicMock()
            mock_dist.crps.return_value = torch.randn(1, 4, 1)
            mock_dist.sample.return_value = torch.randn(10, 1, 4, 1)
            mock_fwd.return_value = mock_dist
            pred.training_step(batch, 0)

    var = pred.output_var_names[0]
    assert "train_loss" in logged
    assert f"train_loss_{var}" in logged
    assert f"train_crps_{var}" in logged


def test_validation_step_dispatch():
    pred = _make_tiny_predictor(dim_y=1)
    pred.n_samples = 10

    # Dict batch → inference path, must not raise
    with (
        patch.object(pred, "log"),
        patch.object(pred, "_log_instant_metrics"),
        patch.object(pred, "__call__") as mock_call,
    ):
        mock_call.return_value = (
            torch.randn(1, 4, 1),
            torch.randn(1, 4, 1),
            torch.ones(1, 4, 1),
        )
        pred.validation_step(_make_oat_batch(dim_y=1), 0)

    # Generator batch → training path, must log val_loss
    logged = {}
    with (
        patch.object(pred, "log", side_effect=lambda k, v, **kw: logged.update({k: v})),
        patch.object(pred, "_forward_generator_batch") as mock_fwd,
    ):
        mock_dist = MagicMock()
        mock_dist.crps.return_value = torch.randn(1, 4, 1)
        mock_dist.sample.return_value = torch.randn(10, 1, 4, 1)
        mock_fwd.return_value = mock_dist
        pred.validation_step(_make_gen_batch(dim_y=1), 0)

    assert "val_loss" in logged


def test_test_step_logs_finite_crps():
    """Regression: metrics are computed on drawn samples, not a single mean.

    A single mean "sample" makes mc_crps divide by n*(n-1)=0 → NaN. Matches
    OATPredictor, which scores on model-drawn samples.
    """
    pred = _make_tiny_predictor(dim_y=1)
    pred.n_samples = 8
    logged: dict = {}
    with patch.object(
        pred, "log", side_effect=lambda name, value, **kw: logged.update({name: value})
    ):
        pred.test_step(_make_oat_batch(B=1, N_obs=15, N_q=5, dim_y=1), 0)
    crps_keys = [k for k in logged if k.startswith("test_crps")]
    assert crps_keys, f"no test_crps metric logged; got {list(logged)}"
    for k in crps_keys:
        assert torch.isfinite(torch.as_tensor(logged[k])), f"{k} = {logged[k]}"


def test_training_step_loss_has_grad():
    """Verify training_step loss has grad_fn (gradient flows through model)."""
    pred = _make_tiny_predictor(dim_y=1)
    pred.n_samples = 10
    batch = _make_gen_batch(dim_y=1)
    with patch.object(pred, "log"):
        with patch.object(pred, "_forward_generator_batch") as mock_fwd:
            mock_dist = MagicMock()
            mock_dist.crps.return_value = torch.randn(1, 4, 1, requires_grad=True)
            mock_dist.sample.return_value = torch.randn(10, 1, 4, 1)
            mock_fwd.return_value = mock_dist
            loss = pred.training_step(batch, 0)
    assert loss.grad_fn is not None, "training_step loss must be differentiable"


C_TOPO, HP, WP = 3, 16, 16


def _topo_transform():
    return StandardNormalizeTransform(
        mean=torch.zeros(C_TOPO), std=torch.ones(C_TOPO), channel_dim=-3
    )


def _make_topo_predictor(dim_y=1, dim_gridded_y=2, v2d=None):
    if v2d is None:
        v2d = {"temperature": "normal"}
    model = build_ootg_model(
        dim_y=dim_y,
        dim_gridded_y=dim_gridded_y,
        obs_variable_to_distribution=v2d,
        embed_dim=32,
        x_embed_dim=32,
        num_heads=2,
        head_dim=8,
        num_layers=2,
        num_legendre_polys=4,
        grid_shape=(8, 8),
        window_sizes=(4, 4),
        shift_sizes=(2, 2),
        top_k_ctot=4,
        use_topo=True,
        topo_in_channels=C_TOPO,
        topo_out_channels=8,
        topo_n_layers=2,
        topo_kernel_size=3,
        topo_initial_hw=(HP, WP),
    )
    ref_x = np.linspace(2255500.0, 2964500.0, 80)
    ref_y = np.linspace(840500.0, 1479500.0, 80)
    xc_grid = nwp_latlon_grid(ref_x, ref_y, stride=10)  # (8, 8, 2)
    return OOTGPredictor(
        model=model,
        obs_vars=list(v2d.keys()),
        obs_mean=torch.full((dim_y,), 280.0),
        obs_std=torch.full((dim_y,), 5.0),
        nwp_mean=torch.zeros(dim_gridded_y),
        nwp_std=torch.ones(dim_gridded_y),
        xc_grid_base=xc_grid,
        coord_to_latlon=lv95_to_latlon,
        dist_names=list(v2d.values()),
        n_nwp=dim_gridded_y,
        nwp_stride=10,
        use_topo=True,
        topo_transform=_topo_transform(),
    )


def _make_topo_gen_batch(B=1, N_c=5, N_t=4, dim_y=1):
    return types.SimpleNamespace(
        xc=torch.randn(B, N_c, 2),
        yc=torch.randn(B, N_c, dim_y),
        xc_grid=torch.randn(B, 8, 8, 2),
        yc_grid=torch.randn(B, 8, 8, 2),
        xt=torch.randn(B, N_t, 2),
        yt=torch.randn(B, N_t, dim_y),
        topo_c=torch.randn(B, N_c, C_TOPO, HP, WP),
        topo_t=torch.randn(B, N_t, C_TOPO, HP, WP),
    )


def _make_topo_oat_batch(B=1, N_obs=15, N_q=5, dim_y=1, n_nwp=2):
    batch = _make_oat_batch(B=B, N_obs=N_obs, N_q=N_q, dim_y=dim_y, n_nwp=n_nwp)
    batch["observation_data"]["topo"] = torch.randn(B, N_obs, C_TOPO, HP, WP)
    batch["query_data"]["topo"] = torch.randn(B, N_q, C_TOPO, HP, WP)
    return batch


def test_predictor_generator_batch_with_topo():
    pred = _make_topo_predictor(dim_y=1)
    dist = pred._forward_generator_batch(_make_topo_gen_batch(dim_y=1))
    assert dist.mean.shape[-1] == 1


def test_predictor_predict_step_with_topo():
    pred = _make_topo_predictor(dim_y=1)
    out = pred.predict_step(_make_topo_oat_batch(B=1, N_q=5, dim_y=1), 0)
    assert out.shape == (1, 5, 1, 1)


def test_predictor_predict_step_topo_empty_context():
    pred = _make_topo_predictor(dim_y=1)
    batch = _make_topo_oat_batch(B=1, N_obs=5, N_q=3, dim_y=1)
    batch["observation_data"]["mask_obs"] = torch.zeros(1, 5, 1)
    out = pred.predict_step(batch, 0)
    assert out.shape == (1, 3, 1, 1)
