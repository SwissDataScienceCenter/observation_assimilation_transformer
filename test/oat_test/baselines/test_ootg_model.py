import sys
from pathlib import Path

import torch

_TNPDIR = Path(__file__).parents[4] / "deps" / "gridded-tnp"
if _TNPDIR.exists() and str(_TNPDIR) not in sys.path:
    sys.path.insert(0, str(_TNPDIR))

from tnp.data.base import OOTGBatch
from oat.baselines.ootg_likelihood import JointMixedDistribution
from oat.baselines.ootg_model import build_ootg_model


def _batch(dim_y, dim_gridded_y):
    B, Nc, Nt = 1, 10, 5
    return OOTGBatch(
        x=torch.randn(B, Nc + Nt, 2),
        y=torch.randn(B, Nc + Nt, dim_y),
        xc=torch.randn(B, Nc, 2),
        yc=torch.randn(B, Nc, dim_y),
        xt=torch.randn(B, Nt, 2),
        yt=torch.randn(B, Nt, dim_y),
        xc_grid=torch.randn(B, 8, 8, 2),
        yc_grid=torch.randn(B, 8, 8, dim_gridded_y),
    )


def _tiny(dim_y=1, dim_gridded_y=2, v2d=None):
    return build_ootg_model(
        dim_y=dim_y, dim_gridded_y=dim_gridded_y,
        obs_variable_to_distribution=v2d,
        embed_dim=32, x_embed_dim=32, num_heads=2, head_dim=8,
        num_layers=2, num_legendre_polys=4, grid_shape=(8, 8),
        window_sizes=(4, 4), shift_sizes=(2, 2), top_k_ctot=4,
    )


def test_forward_normal_only_returns_joint():
    model = _tiny(dim_y=1, v2d={"temperature": "normal"})
    batch = _batch(1, 2)
    dist = model(batch.xc, batch.yc, batch.xc_grid, batch.yc_grid, batch.xt)
    assert isinstance(dist, JointMixedDistribution)
    assert dist.mean.shape == batch.yt.shape


def test_forward_mixed_returns_joint():
    v2d = {"temperature": "normal", "precipitation": "zero_inflated_lognormal"}
    model = _tiny(dim_y=2, dim_gridded_y=4, v2d=v2d)
    batch = _batch(2, 4)
    dist = model(batch.xc, batch.yc, batch.xc_grid, batch.yc_grid, batch.xt)
    assert isinstance(dist, JointMixedDistribution)
    assert dist.mean.shape == batch.yt.shape


def test_nll_loss_normal_finite():
    model = _tiny(dim_y=1, v2d={"temperature": "normal"})
    batch = _batch(1, 2)
    dist = model(batch.xc, batch.yc, batch.xc_grid, batch.yc_grid, batch.xt)
    loss = -dist.log_prob(batch.yt).sum(-1).mean()
    assert loss.ndim == 0 and loss.isfinite()


def test_nll_loss_mixed_finite():
    v2d = {"temperature": "normal", "precipitation": "zero_inflated_lognormal"}
    model = _tiny(dim_y=2, dim_gridded_y=4, v2d=v2d)
    batch = _batch(2, 4)
    # precipitation must be non-negative for ZILN
    batch = OOTGBatch(
        x=batch.x, y=batch.y,
        xc=batch.xc, yc=torch.cat([batch.yc[..., :1], torch.rand(1, 10, 1)], dim=-1),
        xt=batch.xt, yt=torch.cat([batch.yt[..., :1], torch.rand(1, 5, 1)], dim=-1),
        xc_grid=batch.xc_grid, yc_grid=batch.yc_grid,
    )
    dist = model(batch.xc, batch.yc, batch.xc_grid, batch.yc_grid, batch.xt)
    loss = -dist.log_prob(batch.yt).sum(-1).mean()
    assert loss.ndim == 0 and loss.isfinite()


def test_gradients_flow():
    model = _tiny(dim_y=1, v2d={"temperature": "normal"})
    batch = _batch(1, 2)
    dist = model(batch.xc, batch.yc, batch.xc_grid, batch.yc_grid, batch.xt)
    loss = -dist.log_prob(batch.yt).sum(-1).mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
