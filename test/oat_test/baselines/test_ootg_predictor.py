import sys
import types
from pathlib import Path

import pytest
import torch

_TNPDIR = Path(__file__).parents[3] / "deps" / "gridded-tnp"
if _TNPDIR.exists() and str(_TNPDIR) not in sys.path:
    sys.path.insert(0, str(_TNPDIR))

from oat.baselines.ootg_model import build_ootg_model
from oat.baselines.ootg_predictor import OOTGPredictor
from oat.data_module import transform as T
from oat.data_module.field import Field


def _make_field_batch(n_obs=3, n_query=2, h=8, w=8, c_nwp=2, dim_y=1):
    """Build a synthetic field-collated batch with B=1."""

    def _station_field(n, c=dim_y):
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


def _make_predictor(dim_y=1, c_nwp=2, h=8, w=8, stride=2, obs_mean=0.0, **kwargs):
    strided_shape = (h // stride, w // stride)
    model = build_ootg_model(grid_shape=strided_shape, dim_gridded_y=c_nwp, dim_y=dim_y)
    identity_tf = T.NestedTransform({})
    output_tf = T.StandardNormalizeTransform(
        mean=torch.full((dim_y,), obs_mean), std=torch.ones(dim_y)
    )
    predictor = OOTGPredictor(
        model=model,
        roles={"target": "obs", "global_grid": "nwp_grid_field"},
        coord_to_latlon=_identity_coord,
        grid_shape=(h, w),
        nwp_stride=stride,
        output_transform=output_tf,
        input_transform=identity_tf,
        output_var_names=[f"v{i}" for i in range(dim_y)],
        lr=1e-3,
        n_samples=5,
        **kwargs,
    )
    return predictor


def test_training_step_finite_loss():
    predictor = _make_predictor()
    batch = _make_field_batch()
    loss = predictor.training_step(batch, 0)
    assert torch.isfinite(loss), f"training loss is {loss}"


def test_predict_step_output_shape():
    n_query = 4
    predictor = _make_predictor()
    batch = _make_field_batch(n_query=n_query)
    samples = predictor.predict_step(batch, 0)
    # [B, N_query, n_samples, dim_y]
    assert samples.shape == (1, n_query, 5, 1)


def test_predict_step_physical_units():
    predictor = _make_predictor(obs_mean=100.0)
    batch = _make_field_batch()
    out = predictor.predict_step(batch, 0)
    # Temperature-like values near 100, not normalized ~0
    assert out[..., 0].abs().mean() > 10.0


def test_predict_step_n_samples_override():
    predictor = _make_predictor()
    batch = _make_field_batch()
    out = predictor.predict_step(batch, 0, n_samples=10)
    assert out.shape == (1, 2, 10, 1)


def test_validation_step_logs_val_loss():
    from unittest.mock import patch

    predictor = _make_predictor()
    logged = {}

    def capture(key, val, **kwargs):
        logged[key] = val.item() if hasattr(val, "item") else float(val)

    with torch.no_grad(), patch.object(predictor, "log", side_effect=capture):
        predictor.validation_step(_make_field_batch(), 0)
    assert "val_loss" in logged
    assert torch.isfinite(torch.as_tensor(logged["val_loss"]))


def test_test_step_logs_finite_metrics():
    from unittest.mock import patch

    predictor = _make_predictor()
    logged = {}

    def capture(key, val, **kwargs):
        logged[key] = val.item() if hasattr(val, "item") else float(val)

    with torch.no_grad(), patch.object(predictor, "log", side_effect=capture):
        predictor.test_step(_make_field_batch(), 0)
    for key in ("test_crps_v0", "test_point_mae_v0"):
        assert key in logged, f"missing {key}; got {sorted(logged)}"
        assert torch.isfinite(torch.as_tensor(logged[key])), f"{key} = {logged[key]}"


@pytest.mark.parametrize(
    "step_name,prefix",
    [("validation_step", "val"), ("test_step", "test")],
)
def test_metric_mask_excludes_masked_query_row(step_name, prefix):
    from unittest.mock import patch

    predictor = _make_predictor()

    def _run(batch):
        logged = {}

        def capture(key, val, **kwargs):
            logged[key] = val.item() if hasattr(val, "item") else float(val)

        with torch.no_grad(), patch.object(predictor, "log", side_effect=capture):
            getattr(predictor, step_name)(batch, 0)
        return logged

    batch = _make_field_batch(n_query=3)
    query_field = batch["query_data"]["fields"]["obs"]
    # An extreme target value on one row: it must not affect the metrics
    # when that row is masked.
    query_field.values[0, 1, 0] = 1e6
    logged_counted = _run(batch)
    query_field.mask[0, 1, 0] = 0.0
    logged_masked = _run(batch)
    assert (
        logged_counted[f"{prefix}_point_mae_v0"]
        > logged_masked[f"{prefix}_point_mae_v0"]
    )
    assert (
        logged_counted[f"{prefix}_crps_v0"] > logged_masked[f"{prefix}_crps_v0"]
    )


@pytest.mark.parametrize(
    "step_name,loss_key",
    [("training_step", "train_loss"), ("validation_step", "val_loss")],
)
def test_loss_excludes_masked_query_row(step_name, loss_key):
    from unittest.mock import patch

    predictor = _make_predictor()

    def _run(batch):
        logged = {}

        def capture(key, val, **kwargs):
            logged[key] = val.item() if hasattr(val, "item") else float(val)

        with patch.object(predictor, "log", side_effect=capture):
            result = getattr(predictor, step_name)(batch, 0)
        loss = result.item() if result is not None else logged[loss_key]
        return loss, logged[loss_key]

    batch = _make_field_batch(n_query=3)
    query_field = batch["query_data"]["fields"]["obs"]
    # An extreme target value on one row: it must not move the training/
    # validation loss once that row is masked out.
    query_field.values[0, 1, 0] = 1e6
    loss_counted, logged_counted = _run(batch)
    query_field.mask[0, 1, 0] = 0.0
    loss_masked, logged_masked = _run(batch)
    assert loss_counted > loss_masked
    assert logged_counted == pytest.approx(loss_counted)
    assert logged_masked == pytest.approx(loss_masked)


def test_transfer_batch_to_device_preserves_fields():
    predictor = _make_predictor()
    batch = _make_field_batch()
    moved = predictor.transfer_batch_to_device(batch, torch.device("cpu"), 0)
    obs_moved = moved["observation_data"]["fields"]["obs"]
    assert obs_moved.values.shape == (1, 3, 1)
    assert obs_moved.mask is not None
    grid_moved = moved["global_data"]["fields"]["nwp_grid_field"]
    assert grid_moved.values.shape == (1, 64, 2)


class _FakeDataset:
    def __init__(self, dataset_type="cosmo_peakweather", grid_shape=(80, 80)):
        self.config = types.SimpleNamespace(dataset_type=dataset_type)
        self._grid_shape = grid_shape

    def roles(self):
        return {"target": "obs", "global_grid": "nwp_grid_field"}

    def field_coord_sources(self):
        return {"obs": "stations", "nwp_grid_field": "nwp_grid"}

    def grid_shape(self, coord_source):
        return self._grid_shape if coord_source == "nwp_grid" else None


def _make_cfg(**overrides):
    from omegaconf import OmegaConf

    base = {
        "dataset": {"obs_vars": ["temperature"]},
        "nwp_stride": None,
        "lr": 1e-3,
        "n_samples": 7,
        "ablate_obs": True,
        "ablate_grid": False,
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _build_factory_predictor(dataset, cfg):
    from oat.baselines.ootg_predictor import create_ootg_predictor

    model = build_ootg_model(grid_shape=(4, 4), dim_gridded_y=2, dim_y=1)
    output_tf = T.StandardNormalizeTransform(mean=torch.zeros(1), std=torch.ones(1))
    input_tf = T.NestedTransform({})
    return create_ootg_predictor(model, dataset, output_tf, input_tf, cfg)


def test_create_ootg_predictor_factory_cosmo():
    from oat.baselines.coord_utils import lv95_to_latlon

    predictor = _build_factory_predictor(_FakeDataset("cosmo_peakweather"), _make_cfg())
    assert predictor.roles == {"target": "obs", "global_grid": "nwp_grid_field"}
    assert predictor.coord_to_latlon is lv95_to_latlon
    assert predictor.grid_shape == (80, 80)
    assert predictor.nwp_stride == 10
    assert predictor.output_var_names == ["temperature"]
    assert predictor.lr == 1e-3
    assert predictor.n_samples == 7
    assert predictor.ablate_obs is True
    assert predictor.ablate_grid is False


def test_create_ootg_predictor_factory_era5():
    from oat.baselines.coord_utils import lonlat_swap

    predictor = _build_factory_predictor(_FakeDataset("era5_meteostat"), _make_cfg())
    assert predictor.coord_to_latlon is lonlat_swap
    assert predictor.nwp_stride == 4


def test_create_ootg_predictor_factory_stride_override():
    cfg = _make_cfg(nwp_stride=2)
    predictor = _build_factory_predictor(_FakeDataset("cosmo_peakweather"), cfg)
    assert predictor.nwp_stride == 2


def test_create_ootg_predictor_factory_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown dataset_type"):
        _build_factory_predictor(_FakeDataset("windgust"), _make_cfg())


def test_create_ootg_predictor_factory_missing_grid_shape():
    with pytest.raises(ValueError, match="No grid shape"):
        _build_factory_predictor(_FakeDataset(grid_shape=None), _make_cfg())
