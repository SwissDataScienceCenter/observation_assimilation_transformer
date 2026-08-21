import torch

from oat.data_module.field import Field


def _make_batch(n_obs=3, n_query=2, h=4, w=6, c_nwp=2):
    """Build a synthetic collated field batch with B=1."""

    def _station_field(n, c=1):
        coords = torch.randn(1, n, 3)
        vals = torch.randn(1, n, c)
        mask = torch.ones(1, n, c)
        return Field(coords, vals, [f"v{i}" for i in range(c)], mask)

    grid_coords = torch.randn(1, h * w, 3)
    grid_vals = torch.randn(1, h * w, c_nwp)
    grid_field = Field(grid_coords, grid_vals, [f"nwp{i}" for i in range(c_nwp)], None)
    return {
        "index": torch.tensor([0]),
        "observation_data": {"fields": {"obs": _station_field(n_obs)}},
        "query_data": {"fields": {"obs": _station_field(n_query)}},
        "global_data": {"fields": {"nwp_grid_field": grid_field}},
    }


def _identity_coord(c: torch.Tensor) -> torch.Tensor:
    return c


def test_unwrap_shapes():
    from oat.predictor_module.field_batch_utils import _unwrap_fields_to_ootg_batch

    roles = {"target": "obs", "global_grid": "nwp_grid_field"}
    batch = _make_batch(n_obs=3, n_query=2, h=4, w=6, c_nwp=2)
    result = _unwrap_fields_to_ootg_batch(
        batch,
        roles,
        _identity_coord,
        grid_shape=(4, 6),
        nwp_stride=1,
    )
    assert result.xc.shape == (1, 3, 2)
    assert result.yc.shape == (1, 3, 1)
    assert result.xt.shape == (1, 2, 2)
    assert result.yt.shape == (1, 2, 1)
    assert result.xc_grid.shape == (1, 4, 6, 2)
    assert result.yc_grid.shape == (1, 4, 6, 2)
    assert result.x.shape == (1, 5, 2)
    assert result.y.shape == (1, 5, 1)


def test_unwrap_stride():
    from oat.predictor_module.field_batch_utils import _unwrap_fields_to_ootg_batch

    roles = {"target": "obs", "global_grid": "nwp_grid_field"}
    batch = _make_batch(h=8, w=6, c_nwp=1)
    result = _unwrap_fields_to_ootg_batch(
        batch,
        roles,
        _identity_coord,
        grid_shape=(8, 6),
        nwp_stride=2,
    )
    assert result.xc_grid.shape == (1, 4, 3, 2)
    assert result.yc_grid.shape == (1, 4, 3, 1)


def test_unwrap_ablate_obs():
    from oat.predictor_module.field_batch_utils import _unwrap_fields_to_ootg_batch

    roles = {"target": "obs", "global_grid": "nwp_grid_field"}
    batch = _make_batch(n_obs=5)
    result = _unwrap_fields_to_ootg_batch(
        batch,
        roles,
        _identity_coord,
        grid_shape=(4, 6),
        nwp_stride=1,
        ablate_obs=True,
    )
    assert result.xc.shape[1] == 0
    assert result.yc.shape[1] == 0


def test_unwrap_ablate_grid():
    from oat.predictor_module.field_batch_utils import _unwrap_fields_to_ootg_batch

    roles = {"target": "obs", "global_grid": "nwp_grid_field"}
    batch = _make_batch()
    result = _unwrap_fields_to_ootg_batch(
        batch,
        roles,
        _identity_coord,
        grid_shape=(4, 6),
        nwp_stride=1,
        ablate_grid=True,
    )
    assert (result.yc_grid == 0).all()


def test_unwrap_drops_masked_obs():
    from oat.predictor_module.field_batch_utils import _unwrap_fields_to_ootg_batch

    roles = {"target": "obs", "global_grid": "nwp_grid_field"}
    batch = _make_batch(n_obs=4)
    batch["observation_data"]["fields"]["obs"].mask[0, 1, :] = 0.0
    result = _unwrap_fields_to_ootg_batch(
        batch,
        roles,
        _identity_coord,
        grid_shape=(4, 6),
        nwp_stride=1,
    )
    assert result.xc.shape[1] == 3  # 4 - 1 masked = 3
