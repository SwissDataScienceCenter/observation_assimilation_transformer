import math

import matplotlib
import matplotlib.collections
import numpy as np
import pandas as pd
import pytest
import torch

matplotlib.use("Agg")
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt

from oat.plotting_module.plotter import OATPlotter


def _make_predictor(N_q: int = 1, n_samples: int = 10, n_vars: int = 2):
    predictor = MagicMock()
    predictor.var_to_indx = {"temp": 0, "wspd": 1}
    predictor.output_var_names = ["temp", "wspd"]
    predictor._normalize_batch.side_effect = lambda batch: batch
    predictor.predict_samples.return_value = torch.zeros(1, N_q, n_samples, n_vars)
    return predictor


def _make_baseline_predictor(N_q: int = 1, n_samples: int = 10, n_vars: int = 2):
    """Mock that matches NWPInterpolatorPredictor's plotter interface."""
    predictor = MagicMock()
    predictor.var_to_indx = {"temp": 0, "wspd": 1}
    predictor.output_var_names = ["temp", "wspd"]
    predictor._normalize_batch.side_effect = lambda batch: batch
    predictor.predict_samples.return_value = torch.zeros(1, N_q, n_samples, n_vars)
    return predictor


def _make_dataset(n_stations: int = 5, n_times: int = 4, n_vars: int = 2, plot_config: dict | None = None):
    ds = MagicMock()
    ds.station_coords = torch.zeros(n_stations, 2)
    ds.obs_tensor = torch.zeros(n_times, n_stations, n_vars)
    ds.obs_mask_tensor = torch.ones(n_times, n_stations, n_vars)
    ds.times = pd.date_range("2022-01-01", periods=n_times, freq="h", tz="UTC")
    ds.__getitem__.return_value = {
        "global_data": {
            "global_field": torch.zeros(4, 8, 8),
            "time_emb": torch.zeros(4),
        },
        "observation_data": {
            "coords": torch.zeros(n_stations, 2),
            "geo": torch.zeros(n_stations, 3),
            "topo": torch.zeros(n_stations, 1, 32, 32),
            "obs": torch.zeros(n_stations, n_vars),
            "mask_obs": torch.ones(n_stations, n_vars),
        },
    }
    ds.get_query.return_value = {
        "coords": torch.zeros(1, 2),
        "topo": torch.zeros(1, 1, 32, 32),
        "geo": torch.zeros(1, 3),
    }
    config = MagicMock()
    config.obs_plot_config = plot_config
    ds.config = config
    return ds


def test_run_inference_returns_correct_shape():
    N_q, n_samples, n_vars = 3, 10, 2
    predictor = _make_predictor(N_q=N_q, n_samples=n_samples, n_vars=n_vars)
    dataset = _make_dataset()

    plotter = OATPlotter(
        predictor, dataset, obs_station_indices=[0, 1, 2], n_samples=n_samples
    )
    query_coords = torch.zeros(N_q, 2)
    samples = plotter._run_inference(time_idx=0, query_coords=query_coords)

    assert samples.shape == (N_q, n_samples, n_vars)


def test_plotter_stores_n_samples():
    predictor = _make_predictor()
    dataset = _make_dataset()
    assert OATPlotter(predictor, dataset, n_samples=42).n_samples == 42


def test_plotter_default_n_samples():
    predictor = _make_predictor()
    dataset = _make_dataset()
    assert OATPlotter(predictor, dataset).n_samples == 100


def test_run_inference_with_baseline_predictor():
    """OATPlotter works with a mock satisfying the BasePredictor interface."""
    N_q, n_samples, n_vars = 3, 10, 2
    predictor = _make_baseline_predictor(N_q=N_q, n_samples=n_samples, n_vars=n_vars)
    dataset = _make_dataset()

    plotter = OATPlotter(predictor, dataset, obs_station_indices=[0, 1, 2], n_samples=n_samples)
    samples = plotter._run_inference(time_idx=0, query_coords=torch.zeros(N_q, 2))

    assert samples.shape == (N_q, n_samples, n_vars)
    predictor._normalize_batch.assert_called_once()
    predictor.predict_samples.assert_called_once()
    assert predictor.predict_samples.call_args[0][1] == n_samples


def test_plot_timeseries_draws_line_and_band():
    predictor = _make_predictor(N_q=1, n_samples=10)
    dataset = _make_dataset(n_stations=5, n_times=4)
    plotter = OATPlotter(predictor, dataset, obs_station_indices=[0, 1, 2])

    fig, ax = plt.subplots()
    fixed_samples = torch.randn(1, 10, 2)  # [N_q=1, n_samples, n_vars]
    with patch.object(plotter, "_run_inference", return_value=fixed_samples):
        plotter.plot_timeseries(
            ax=ax, center=(8.0, 47.0), var="temp", time_indices=[0, 1, 2, 3]
        )

    assert len(ax.lines) >= 1, "Expected at least one line (mean)"
    assert (
        len(ax.collections) >= 1
    ), "Expected at least one collection (uncertainty band)"
    plt.close(fig)


def _spatial_plotter(extent=0.4, resolution=0.2):
    """Returns a plotter where _run_inference is patched to the right N_q."""
    # With extent=0.4, resolution=0.2:
    #   axis0: center-0.4 … center+0.4, step 0.2  → 5 values
    #   axis1: center-0.4 … center+0.4, step 0.2  → 5 values  → N_q = 25
    predictor = _make_predictor(N_q=25, n_samples=10)
    dataset = _make_dataset()
    return OATPlotter(predictor, dataset, obs_station_indices=[0, 1, 2])


def test_plot_spatial_snapshot_returns_quadmesh():
    plotter = _spatial_plotter()
    fig, ax = plt.subplots()

    # 5 x 5 = 25 query points, 2 vars
    fixed_samples = torch.randn(25, 10, 2)
    with patch.object(plotter, "_run_inference", return_value=fixed_samples):
        mesh = plotter.plot_spatial_snapshot(
            ax=ax,
            var="temp",
            time_idx=0,
            center=(8.0, 47.0),
            extent=0.4,
            resolution=0.2,
            plot_type="mean",
        )

    assert mesh is not None
    plt.close(fig)


def test_plot_spatial_snapshot_std():
    plotter = _spatial_plotter()
    fig, ax = plt.subplots()

    fixed_samples = torch.randn(25, 10, 2)
    with patch.object(plotter, "_run_inference", return_value=fixed_samples):
        mesh = plotter.plot_spatial_snapshot(
            ax=ax,
            var="wspd",
            time_idx=0,
            center=(8.0, 47.0),
            extent=0.4,
            resolution=0.2,
            plot_type="std",
        )

    assert mesh is not None
    plt.close(fig)


def test_plot_spatial_snapshot_invalid_plot_type():
    plotter = _spatial_plotter()
    fig, ax = plt.subplots()
    fixed_samples = torch.randn(25, 10, 2)
    with patch.object(plotter, "_run_inference", return_value=fixed_samples):
        with pytest.raises(ValueError, match="plot_type"):
            plotter.plot_spatial_snapshot(
                ax=ax,
                var="temp",
                time_idx=0,
                center=(8.0, 47.0),
                extent=0.4,
                resolution=0.2,
                plot_type="median",  # invalid
            )
    plt.close(fig)


def test_plot_spatial_snapshot_at_station_delegates_to_plot_spatial_snapshot():
    predictor = _make_predictor(N_q=25, n_samples=10)
    dataset = _make_dataset(n_stations=5, n_times=4)
    dataset.station_coords = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    )
    plotter = OATPlotter(predictor, dataset, obs_station_indices=[0, 1, 2])

    fig, ax = plt.subplots()
    sentinel = object()
    with patch.object(
        plotter, "plot_spatial_snapshot", return_value=sentinel
    ) as mock_sp:
        result = plotter.plot_spatial_snapshot_at_station(
            ax=ax,
            var="temp",
            time_idx=0,
            station_idx=3,
            extent=0.4,
            resolution=0.2,
            plot_type="std",
        )
    mock_sp.assert_called_once_with(
        ax,
        "temp",
        0,
        (7.0, 8.0),
        extent=0.4,
        resolution=0.2,
        plot_type="std",
        show_stations=True,
    )
    assert result is sentinel
    plt.close(fig)


def test_plotter_importable_from_package():
    from oat.plotting_module import OATPlotter as _OATPlotter

    assert _OATPlotter is OATPlotter


def test_plot_timeseries_at_station_delegates_to_plot_timeseries():
    predictor = _make_predictor(N_q=1, n_samples=10)
    dataset = _make_dataset(n_stations=5, n_times=4)
    dataset.station_coords = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    )
    plotter = OATPlotter(predictor, dataset, obs_station_indices=[0, 1, 2])

    fig, ax = plt.subplots()
    with patch.object(plotter, "plot_timeseries") as mock_ts:
        plotter.plot_timeseries_at_station(
            ax=ax, station_idx=2, var="temp", time_indices=[0, 1]
        )
    mock_ts.assert_called_once_with(
        ax, (5.0, 6.0), "temp", [0, 1], quantile_lo=0.05, quantile_hi=0.95
    )
    plt.close(fig)


def test_plot_timeseries_at_station_plots_observations():
    predictor = _make_predictor(N_q=1, n_samples=10)
    dataset = _make_dataset(n_stations=3, n_times=4, n_vars=2)
    dataset.station_coords = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # Set known observation values for station 1, var "temp" (index 0)
    dataset.obs_tensor = torch.zeros(4, 3, 2)
    dataset.obs_tensor[:, 1, 0] = torch.tensor([10.0, 20.0, 30.0, 40.0])
    dataset.obs_mask_tensor = torch.ones(4, 3, 2)  # all observed
    plotter = OATPlotter(predictor, dataset, obs_station_indices=[0, 1, 2])

    fig, ax = plt.subplots()
    with patch.object(plotter, "plot_timeseries"):
        plotter.plot_timeseries_at_station(
            ax=ax, station_idx=1, var="temp", time_indices=[0, 1, 2, 3]
        )

    # A scatter call adds a PathCollection to ax.collections
    scatter_collections = [c for c in ax.collections]
    assert len(scatter_collections) >= 1, "Expected at least one scatter (observations)"
    plt.close(fig)


def test_plot_timeseries_at_station_masks_missing_observations():
    predictor = _make_predictor(N_q=1, n_samples=10)
    dataset = _make_dataset(n_stations=3, n_times=4, n_vars=2)
    dataset.station_coords = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    dataset.obs_tensor = torch.zeros(4, 3, 2)
    dataset.obs_tensor[:, 0, 0] = torch.tensor([10.0, 20.0, 30.0, 40.0])
    # Mark time indices 1 and 3 as missing for station 0, var 0
    dataset.obs_mask_tensor = torch.ones(4, 3, 2)
    dataset.obs_mask_tensor[1, 0, 0] = 0.0
    dataset.obs_mask_tensor[3, 0, 0] = 0.0
    plotter = OATPlotter(predictor, dataset, obs_station_indices=[0, 1, 2])

    fig, ax = plt.subplots()
    with patch.object(plotter, "plot_timeseries"):
        plotter.plot_timeseries_at_station(
            ax=ax, station_idx=0, var="temp", time_indices=[0, 1, 2, 3]
        )

    # Retrieve the y-data of the scatter
    scatter_ys = list(ax.collections[0].get_offsets()[:, 1])
    assert math.isnan(scatter_ys[1]), "Missing observation at t=1 should be NaN"
    assert math.isnan(scatter_ys[3]), "Missing observation at t=3 should be NaN"
    assert not math.isnan(scatter_ys[0])
    assert not math.isnan(scatter_ys[2])
    plt.close(fig)


# --- station overlay tests ---


def _path_collections(ax):
    return [
        c
        for c in ax.collections
        if isinstance(c, matplotlib.collections.PathCollection)
    ]


def test_plot_spatial_snapshot_stations_default():
    """show_stations=True (default) adds station and POI PathCollections to the axes."""
    plotter = _spatial_plotter()
    fig, ax = plt.subplots()
    fixed_samples = torch.randn(25, 10, 2)
    with patch.object(plotter, "_run_inference", return_value=fixed_samples):
        plotter.plot_spatial_snapshot(
            ax=ax,
            var="temp",
            time_idx=0,
            center=(8.0, 47.0),
            extent=0.4,
            resolution=0.2,
        )
    # 2 station groups (observed circles, not-observed squares) + 1 POI marker
    assert len(_path_collections(ax)) == 3, "Expected two station groups + POI"
    plt.close(fig)


def test_plot_spatial_snapshot_no_stations():
    """show_stations=False leaves no PathCollection on the axes."""
    plotter = _spatial_plotter()
    fig, ax = plt.subplots()
    fixed_samples = torch.randn(25, 10, 2)
    with patch.object(plotter, "_run_inference", return_value=fixed_samples):
        plotter.plot_spatial_snapshot(
            ax=ax,
            var="temp",
            time_idx=0,
            center=(8.0, 47.0),
            extent=0.4,
            resolution=0.2,
            show_stations=False,
        )
    assert len(_path_collections(ax)) == 0
    plt.close(fig)


def test_plot_spatial_snapshot_stations_observed_vs_not():
    """Correct station counts in each group.

    obs_station_indices=[0, 1, 2], n_stations=5.
    At time_idx=0, var='temp' (idx 0): station 1 mask=0 → not observed.
    Expected: observed={0, 2} (2 pts), not_observed={1, 3, 4} (3 pts).
    """
    predictor = _make_predictor(N_q=25, n_samples=10)
    dataset = _make_dataset(n_stations=5, n_times=4, n_vars=2)
    # Mark station 1 as missing at time 0, var 0
    dataset.obs_mask_tensor = torch.ones(4, 5, 2)
    dataset.obs_mask_tensor[0, 1, 0] = 0.0
    plotter = OATPlotter(predictor, dataset, obs_station_indices=[0, 1, 2])

    fig, ax = plt.subplots()
    fixed_samples = torch.randn(25, 10, 2)
    with patch.object(plotter, "_run_inference", return_value=fixed_samples):
        plotter.plot_spatial_snapshot(
            ax=ax,
            var="temp",
            time_idx=0,
            center=(0.0, 0.0),
            extent=0.4,
            resolution=0.2,
        )

    pcs = _path_collections(ax)
    # Exclude POI (zorder=6); station groups use zorder=5
    station_pcs = [pc for pc in pcs if pc.get_zorder() < 6]
    assert len(station_pcs) == 2
    observed_pc, not_observed_pc = station_pcs
    assert len(observed_pc.get_offsets()) == 2, "Stations 0 and 2 should be observed"
    assert (
        len(not_observed_pc.get_offsets()) == 3
    ), "Stations 1, 3, 4 should be not observed"
    plt.close(fig)


def test_plot_spatial_snapshot_sets_axis_limits():
    """Axis limits are clamped to the grid extent after plotting."""
    plotter = _spatial_plotter(extent=0.4, resolution=0.2)
    fig, ax = plt.subplots()
    fixed_samples = torch.randn(25, 10, 2)
    center = (8.0, 47.0)
    extent = 0.4
    with patch.object(plotter, "_run_inference", return_value=fixed_samples):
        plotter.plot_spatial_snapshot(
            ax=ax,
            var="temp",
            time_idx=0,
            center=center,
            extent=extent,
            resolution=0.2,
        )
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert abs(xlim[0] - (center[0] - extent)) < 1e-6
    assert abs(xlim[1] - (center[0] + extent)) < 1e-6
    assert abs(ylim[0] - (center[1] - extent)) < 1e-6
    assert abs(ylim[1] - (center[1] + extent)) < 1e-6
    plt.close(fig)


def test_plot_spatial_snapshot_legend_shown():
    """A legend is added when show_stations=True."""
    plotter = _spatial_plotter()
    fig, ax = plt.subplots()
    fixed_samples = torch.randn(25, 10, 2)
    with patch.object(plotter, "_run_inference", return_value=fixed_samples):
        plotter.plot_spatial_snapshot(
            ax=ax,
            var="temp",
            time_idx=0,
            center=(8.0, 47.0),
            extent=0.4,
            resolution=0.2,
        )
    assert ax.get_legend() is not None
    plt.close(fig)


def test_plot_spatial_snapshot_observed_colored_in_mean_mode():
    """Observed station markers are colored by obs value (not white) in mean mode."""
    predictor = _make_predictor(N_q=25, n_samples=10)
    dataset = _make_dataset(n_stations=3, n_times=4, n_vars=2)
    # Give obs_tensor a non-zero value so the color is distinguishable
    dataset.obs_tensor = torch.zeros(4, 3, 2)
    dataset.obs_tensor[0, 0, 0] = 5.0
    dataset.obs_mask_tensor = torch.ones(4, 3, 2)
    plotter = OATPlotter(predictor, dataset, obs_station_indices=[0, 1, 2])

    fig, ax = plt.subplots()
    fixed_samples = torch.randn(25, 10, 2)
    with patch.object(plotter, "_run_inference", return_value=fixed_samples):
        plotter.plot_spatial_snapshot(
            ax=ax,
            var="temp",
            time_idx=0,
            center=(0.0, 0.0),
            extent=0.4,
            resolution=0.2,
            plot_type="mean",
        )

    observed_pc = _path_collections(ax)[0]
    facecolors = observed_pc.get_facecolor()
    # With obs values [5.0, 0.0, 0.0] and a non-trivial field, colors should not all
    # be identical
    assert facecolors.shape[0] == 3, "Three observed stations"
    # Stations have different obs values so their RGBA should differ
    assert not (
        facecolors[0] == facecolors[1]
    ).all(), "Different obs values → different colors"
    plt.close(fig)


def test_plotter_stores_eval_indices():
    plotter = OATPlotter(
        predictor=_make_predictor(),
        dataset=_make_dataset(),
        eval_indices=[3, 4],
    )
    assert plotter.eval_indices == [3, 4]


def test_plotter_default_eval_indices_empty():
    plotter = OATPlotter(predictor=_make_predictor(), dataset=_make_dataset())
    assert plotter.eval_indices == []


def test_plot_spatial_snapshot_eval_all_observed():
    """eval_indices mode, all obs have mask: 2 station groups
    (colored circles + eval squares).
    """
    plotter = OATPlotter(
        predictor=_make_predictor(N_q=25, n_samples=10),
        dataset=_make_dataset(),  # all masks = 1
        obs_station_indices=[0, 1, 2],
        eval_indices=[3, 4],
    )
    fig, ax = plt.subplots()
    with patch.object(plotter, "_run_inference", return_value=torch.zeros(25, 10, 2)):
        plotter.plot_spatial_snapshot(
            ax, "temp", 0, (0.0, 0.0), extent=0.4, resolution=0.2
        )
    pcs = _path_collections(ax)
    # Collect groups by non-zero point count (skip POI which is 1 point at center)
    station_counts = sorted(
        len(pc.get_offsets()) for pc in pcs if len(pc.get_offsets()) > 1
    )
    assert station_counts == [2, 3], "2 eval squares + 3 colored obs circles"
    plt.close(fig)


def test_plot_spatial_snapshot_eval_with_missing_obs():
    """eval_indices mode, one obs missing: 3 station groups."""
    dataset = _make_dataset()
    dataset.obs_mask_tensor[0, 1, 0] = 0.0  # station 1 missing at time 0, var 0
    plotter = OATPlotter(
        predictor=_make_predictor(N_q=25, n_samples=10),
        dataset=dataset,
        obs_station_indices=[0, 1, 2],
        eval_indices=[3, 4],
    )
    fig, ax = plt.subplots()
    with patch.object(plotter, "_run_inference", return_value=torch.zeros(25, 10, 2)):
        plotter.plot_spatial_snapshot(
            ax, "temp", 0, (0.0, 0.0), extent=0.4, resolution=0.2
        )
    pcs = _path_collections(ax)
    # Filter out POI (zorder=6); station groups use zorder=5
    station_counts = sorted(len(pc.get_offsets()) for pc in pcs if pc.get_zorder() < 6)
    # 2 colored circles (obs 0,2) + 1 white circle (missing obs 1)
    # + 2 eval squares (3,4)
    assert station_counts == [
        1,
        2,
        2,
    ], "1 missing-obs circle + 2 obs circles + 2 eval squares"
    plt.close(fig)


# --- plot_spatial_difference tests ---


def _shape_aware_samples(time_idx, query_coords, predictor=None):
    """Side-effect for _run_inference that returns the correct N_q shape."""
    n_q = query_coords.shape[0]
    return torch.randn(n_q, 10, 2)


def test_plot_spatial_difference_returns_quadmesh():
    plotter = _spatial_plotter()
    other = _make_predictor(N_q=25, n_samples=10)
    fig, ax = plt.subplots()
    with patch.object(plotter, "_run_inference", side_effect=_shape_aware_samples):
        mesh = plotter.plot_spatial_difference(
            other,
            ax=ax,
            var="temp",
            time_idx=0,
            center=(8.0, 47.0),
            extent=0.4,
            resolution=0.2,
        )
    assert mesh is not None
    plt.close(fig)


def test_plot_spatial_difference_axis_limits():
    plotter = _spatial_plotter()
    other = _make_predictor(N_q=25, n_samples=10)
    fig, ax = plt.subplots()
    center = (8.0, 47.0)
    extent = 0.4
    with patch.object(plotter, "_run_inference", side_effect=_shape_aware_samples):
        plotter.plot_spatial_difference(
            other,
            ax=ax,
            var="temp",
            time_idx=0,
            center=center,
            extent=extent,
            resolution=0.2,
        )
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert abs(xlim[0] - (center[0] - extent)) < 1e-6
    assert abs(xlim[1] - (center[0] + extent)) < 1e-6
    assert abs(ylim[0] - (center[1] - extent)) < 1e-6
    assert abs(ylim[1] - (center[1] + extent)) < 1e-6
    plt.close(fig)


def test_plot_spatial_difference_no_stations():
    plotter = _spatial_plotter()
    other = _make_predictor(N_q=25, n_samples=10)
    fig, ax = plt.subplots()
    with patch.object(plotter, "_run_inference", return_value=torch.randn(25, 10, 2)):
        plotter.plot_spatial_difference(
            other,
            ax=ax,
            var="temp",
            time_idx=0,
            center=(8.0, 47.0),
            extent=0.4,
            resolution=0.2,
            show_stations=False,
        )
    assert len(_path_collections(ax)) == 0
    plt.close(fig)


def test_plot_spatial_difference_stations_shown():
    """show_stations=True adds observed circles, not-observed squares, and POI marker."""
    plotter = _spatial_plotter()
    other = _make_predictor(N_q=25, n_samples=10)
    fig, ax = plt.subplots()
    with patch.object(plotter, "_run_inference", side_effect=_shape_aware_samples):
        plotter.plot_spatial_difference(
            other,
            ax=ax,
            var="temp",
            time_idx=0,
            center=(8.0, 47.0),
            extent=0.4,
            resolution=0.2,
            show_stations=True,
        )
    # 2 station groups (observed circles, not-observed squares) + 1 POI marker
    assert len(_path_collections(ax)) == 3
    plt.close(fig)


def test_plot_spatial_difference_calls_both_predictors():
    """_run_inference is called twice for the grid (self and other predictor)."""
    plotter = _spatial_plotter()
    other = _make_predictor(N_q=25, n_samples=10)
    fig, ax = plt.subplots()

    def _side_effect(time_idx, query_coords, predictor=None):
        n_q = query_coords.shape[0]
        return torch.zeros(n_q, 10, 2) if predictor is not None else torch.ones(n_q, 10, 2)

    with patch.object(plotter, "_run_inference", side_effect=_side_effect) as mock_inf:
        plotter.plot_spatial_difference(
            other,
            ax=ax,
            var="temp",
            time_idx=0,
            center=(8.0, 47.0),
            extent=0.4,
            resolution=0.2,
            show_stations=False,  # isolate grid calls from station call
        )

    # Without stations: exactly 2 calls (grid for self, grid for other)
    assert mock_inf.call_count == 2
    # Second call must have passed the other predictor
    second_call_kwargs = mock_inf.call_args_list[1].kwargs
    assert second_call_kwargs.get("predictor") is other
    plt.close(fig)


def test_plot_spatial_difference_at_station_delegates():
    predictor = _make_predictor(N_q=25, n_samples=10)
    dataset = _make_dataset(n_stations=5, n_times=4)
    dataset.station_coords = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    )
    plotter = OATPlotter(predictor, dataset, obs_station_indices=[0, 1, 2])
    other = _make_predictor(N_q=25, n_samples=10)

    fig, ax = plt.subplots()
    sentinel = object()
    with patch.object(plotter, "plot_spatial_difference", return_value=sentinel) as mock_diff:
        result = plotter.plot_spatial_difference_at_station(
            other,
            ax=ax,
            var="temp",
            time_idx=0,
            station_idx=3,
            extent=0.4,
            resolution=0.2,
        )
    mock_diff.assert_called_once_with(
        other,
        ax,
        "temp",
        0,
        (7.0, 8.0),
        extent=0.4,
        resolution=0.2,
        show_stations=True,
    )
    assert result is sentinel
    plt.close(fig)


# ── _overlay_stations ──────────────────────────────────────────────────────

class TestOverlayStations:
    def setup_method(self):
        matplotlib.use("Agg")

    def _make_plotter(self, obs_indices=None, eval_indices=None, n_stations=5):
        predictor = _make_predictor()
        dataset = _make_dataset(n_stations=n_stations)
        dataset.station_coords = torch.tensor(
            [[float(i), float(i)] for i in range(n_stations)]
        )
        dataset.obs_mask_tensor = torch.ones(4, n_stations, 2)
        dataset.obs_tensor = torch.ones(4, n_stations, 2) * 2.0
        return OATPlotter(
            predictor, dataset,
            obs_station_indices=obs_indices,
            eval_indices=eval_indices,
        )

    def test_observed_stations_rendered_as_circles(self):
        plotter = self._make_plotter(obs_indices=[0, 1], eval_indices=[])
        fig, ax = plt.subplots()
        plotter._overlay_stations(ax, time_idx=0, var_idx=0)
        assert any(c.get_label() == "observed" for c in ax.collections)
        plt.close(fig)

    def test_missing_obs_no_eval_set_renders_as_square(self):
        plotter = self._make_plotter(obs_indices=[0, 1], eval_indices=[])
        plotter.dataset.obs_mask_tensor[0, 0, 0] = 0  # station 0 missing at t=0
        fig, ax = plt.subplots()
        plotter._overlay_stations(ax, time_idx=0, var_idx=0)
        labels = [c.get_label() for c in ax.collections]
        assert "not observed" in labels
        plt.close(fig)

    def test_eval_stations_rendered_as_squares(self):
        plotter = self._make_plotter(obs_indices=[0, 1], eval_indices=[2, 3])
        fig, ax = plt.subplots()
        plotter._overlay_stations(ax, time_idx=0, var_idx=0)
        labels = [c.get_label() for c in ax.collections]
        assert "eval observed" in labels
        plt.close(fig)

    def test_missing_obs_with_eval_set_renders_as_missing(self):
        plotter = self._make_plotter(obs_indices=[0, 1], eval_indices=[2])
        plotter.dataset.obs_mask_tensor[0, 0, 0] = 0
        fig, ax = plt.subplots()
        plotter._overlay_stations(ax, time_idx=0, var_idx=0)
        labels = [c.get_label() for c in ax.collections]
        assert "missing" in labels
        plt.close(fig)

    def test_missing_eval_renders_as_eval_missing(self):
        plotter = self._make_plotter(obs_indices=[0], eval_indices=[1])
        plotter.dataset.obs_mask_tensor[0, 1, 0] = 0
        fig, ax = plt.subplots()
        plotter._overlay_stations(ax, time_idx=0, var_idx=0)
        labels = [c.get_label() for c in ax.collections]
        assert "eval missing" in labels
        plt.close(fig)

    def test_mesh_colors_obs_stations(self):
        plotter = self._make_plotter(obs_indices=[0, 1], eval_indices=[])
        fig, ax = plt.subplots()
        mesh = ax.pcolormesh([[0, 1], [2, 3]])
        plotter._overlay_stations(ax, time_idx=0, var_idx=0, mesh=mesh)
        obs_scatter = next(c for c in ax.collections if c.get_label() == "observed")
        # When mesh is provided, facecolors should be RGBA arrays (not a single string)
        assert obs_scatter.get_facecolor().shape[1] == 4
        plt.close(fig)

    def test_color_by_overrides_obs_values(self):
        plotter = self._make_plotter(obs_indices=[0, 1], eval_indices=[])
        plotter.dataset.obs_tensor[0, 0, 0] = 999.0
        fig, ax = plt.subplots()
        mesh = ax.pcolormesh([[0, 1], [2, 3]])
        plotter._overlay_stations(ax, time_idx=0, var_idx=0, mesh=mesh, color_by={0: 1.0, 1: 2.0})
        obs_scatter = next(c for c in ax.collections if c.get_label() == "obs − NWP")
        assert obs_scatter is not None
        plt.close(fig)


# ── plot_domain_overview ───────────────────────────────────────────────────

import matplotlib.figure


class TestPlotDomainOverview:
    def setup_method(self):
        matplotlib.use("Agg")

    def _make_plotter_and_other(self, n_samples=5, n_vars=2):
        n_stations = 3
        predictor = _make_predictor(N_q=4, n_samples=n_samples, n_vars=n_vars)
        other = _make_baseline_predictor(N_q=4, n_samples=n_samples, n_vars=n_vars)
        dataset = _make_dataset(n_stations=n_stations)
        dataset.station_coords = torch.zeros(n_stations, 2)
        dataset.obs_mask_tensor = torch.ones(4, n_stations, n_vars)
        dataset.obs_tensor = torch.ones(4, n_stations, n_vars)
        plotter = OATPlotter(
            predictor, dataset,
            obs_station_indices=[0, 1],
            n_samples=n_samples,
        )
        return plotter, other

    def test_returns_figure(self):
        plotter, other = self._make_plotter_and_other()
        xs = np.linspace(0, 1, 2)
        ys = np.linspace(0, 1, 2)
        with patch.object(plotter, "_run_inference") as mock_inf:
            mock_inf.return_value = torch.zeros(4, 5, 2)
            fig = plotter.plot_domain_overview(other, "temp", 0, xs, ys)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_figure_has_four_axes(self):
        plotter, other = self._make_plotter_and_other()
        xs = np.linspace(0, 1, 2)
        ys = np.linspace(0, 1, 2)
        with patch.object(plotter, "_run_inference") as mock_inf:
            mock_inf.return_value = torch.zeros(4, 5, 2)
            fig = plotter.plot_domain_overview(other, "temp", 0, xs, ys)
        assert len(fig.axes) >= 4
        plt.close(fig)

    def test_title_is_set_when_provided(self):
        plotter, other = self._make_plotter_and_other()
        xs = np.linspace(0, 1, 2)
        ys = np.linspace(0, 1, 2)
        with patch.object(plotter, "_run_inference") as mock_inf:
            mock_inf.return_value = torch.zeros(4, 5, 2)
            fig = plotter.plot_domain_overview(other, "temp", 0, xs, ys, title="My Title")
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == "My Title"
        plt.close(fig)

    def test_no_suptitle_when_title_empty(self):
        plotter, other = self._make_plotter_and_other()
        xs = np.linspace(0, 1, 2)
        ys = np.linspace(0, 1, 2)
        with patch.object(plotter, "_run_inference") as mock_inf:
            mock_inf.return_value = torch.zeros(4, 5, 2)
            fig = plotter.plot_domain_overview(other, "temp", 0, xs, ys)
        assert fig._suptitle is None or fig._suptitle.get_text() == ""
        plt.close(fig)

    def test_run_inference_called_at_least_twice(self):
        plotter, other = self._make_plotter_and_other()
        xs = np.linspace(0, 1, 2)
        ys = np.linspace(0, 1, 2)
        with patch.object(plotter, "_run_inference") as mock_inf:
            mock_inf.return_value = torch.zeros(4, 5, 2)
            plotter.plot_domain_overview(other, "temp", 0, xs, ys)
        assert mock_inf.call_count >= 2
        plt.close("all")

    def test_second_panel_shares_nwp_colormap(self):
        plotter, other = self._make_plotter_and_other()
        xs = np.linspace(0, 1, 2)
        ys = np.linspace(0, 1, 2)
        with patch.object(plotter, "_run_inference") as mock_inf:
            mock_inf.return_value = torch.zeros(4, 5, 2)
            fig = plotter.plot_domain_overview(other, "temp", 0, xs, ys)
        ax0, ax1 = fig.axes[0], fig.axes[1]
        mesh0 = next(c for c in ax0.get_children() if hasattr(c, "get_clim"))
        mesh1 = next(c for c in ax1.get_children() if hasattr(c, "get_clim"))
        assert mesh0.get_clim() == mesh1.get_clim()
        plt.close(fig)


# ── _get_var_plot_config ───────────────────────────────────────────────────

def test_get_var_plot_config_returns_empty_when_no_config():
    plotter = OATPlotter(_make_predictor(), _make_dataset(plot_config=None))
    assert plotter._get_var_plot_config("temperature") == {}


def test_get_var_plot_config_returns_empty_for_unknown_var():
    plotter = OATPlotter(_make_predictor(), _make_dataset(plot_config={"precip": {"cmap": "Blues"}}))
    assert plotter._get_var_plot_config("temperature") == {}


def test_get_var_plot_config_returns_entry_for_known_var():
    plotter = OATPlotter(
        _make_predictor(),
        _make_dataset(plot_config={"temperature": {"color": "C1", "cmap": "RdBu_r", "vmin": -5.0}}),
    )
    cfg = plotter._get_var_plot_config("temperature")
    assert cfg.get("color") == "C1"
    assert cfg.get("cmap") == "RdBu_r"
    assert cfg.get("vmin") == -5.0
    assert cfg.get("vmax") is None
