import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.plot_extreme_events import (
    EVENTS,
    _daily_peak_means,
    _parse_time,
    _rank_stations,
    _station_window_obs,
    _validate_event,
    _window_sum_field,
    _window_time_indices,
)


def _hourly_times(start="2024-06-28T00", periods=100):
    return pd.date_range(start, periods=periods, freq="h", tz="UTC")


def _event(**overrides):
    return dataclasses.replace(EVENTS[0], **overrides)


def test_window_time_indices_returns_consecutive_indices():
    times = _hourly_times()
    assert _window_time_indices(times, "2024-06-28:12", 3) == [12, 13, 14]


def test_window_time_indices_raises_on_missing_hour():
    times = _hourly_times().delete(13)
    with pytest.raises(ValueError, match="not in the dataset times"):
        _window_time_indices(times, "2024-06-28:12", 3)


def test_window_time_indices_raises_after_dataset_end():
    times = _hourly_times(periods=10)
    with pytest.raises(ValueError, match="not in the dataset times"):
        _window_time_indices(times, "2024-06-28:08", 5)


def test_validate_event_passes_for_held_out_stations_in_test_period():
    event = _event(stations=(185, 164))
    _validate_event(event, [164, 185, 7], range(0, 100), [12, 13], [10, 11, 12, 13])


def test_validate_event_raises_for_station_not_held_out():
    event = _event(stations=(185, 164))
    with pytest.raises(ValueError, match=r"not held-out test stations: \[164\]"):
        _validate_event(event, [185], range(0, 100), [12], [12])


def test_validate_event_raises_for_time_outside_test_period():
    event = _event(stations=(185,))
    with pytest.raises(ValueError, match="outside the test period"):
        _validate_event(event, [185], range(50, 100), [60], [49, 50])


@pytest.mark.parametrize("event", EVENTS, ids=lambda e: e.dataset)
def test_events_are_well_formed(event):
    assert 1 <= len(event.stations) <= 2
    assert event.map_hours >= 1
    assert event.extent > 0
    ts0 = _parse_time(event.ts_start)
    ts1 = ts0 + pd.Timedelta(hours=event.ts_hours)
    m0 = _parse_time(event.map_start)
    m1 = m0 + pd.Timedelta(hours=event.map_hours)
    assert ts0 <= m0 and m1 <= ts1


def _obs_and_mask():
    # 4 hours, 3 stations, 2 variables. Station 1 misses hour 2 of var 0,
    # and its masked slot holds NaN (as in the real obs tensor).
    obs = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)
    mask = torch.ones(4, 3, 2, dtype=torch.bool)
    mask[2, 1, 0] = False
    obs[2, 1, 0] = float("nan")
    return obs, mask


def test_station_window_obs_sums_full_windows():
    obs, mask = _obs_and_mask()
    sums = _station_window_obs(obs, mask, [0, 1, 2, 3], var_idx=0)
    # station 0, var 0 holds 0, 6, 12, 18; station 2 holds 4, 10, 16, 22
    assert sums[0] == 36.0
    assert sums[2] == 52.0


def test_station_window_obs_is_nan_for_a_missing_hour():
    obs, mask = _obs_and_mask()
    sums = _station_window_obs(obs, mask, [0, 1, 2, 3], var_idx=0)
    assert np.isnan(sums[1])
    # the gap is at hour 2 only, so a window without it is full
    assert _station_window_obs(obs, mask, [0, 1], var_idx=0)[1] == 2.0 + 8.0


class _StubPlotter:
    """predict_samples returns (time_idx + query index) for all samples."""

    def __init__(self):
        self.calls = []

    def predict_samples(self, query_coords, time_idx, predictor=None, stage="oat"):
        self.calls.append((time_idx, predictor, stage))
        n_q = query_coords.shape[0]
        base = torch.arange(n_q, dtype=torch.float32) + time_idx
        # [N_q, n_samples=3, n_vars=2]; var 1 is base, var 0 is zero
        samples = torch.zeros(n_q, 3, 2)
        samples[:, :, 1] = base[:, None]
        return samples


def test_window_sum_field_sums_hourly_means():
    plotter = _StubPlotter()
    grid = torch.zeros(6, 2)
    field = _window_sum_field(plotter, None, grid, [10, 11], var_idx=1, shape=(2, 3))
    expected = (np.arange(6) + 10 + np.arange(6) + 11).reshape(2, 3)
    np.testing.assert_allclose(field, expected)
    assert plotter.calls == [(10, None, "oat"), (11, None, "oat")]


def test_window_sum_field_single_hour_is_the_field():
    plotter = _StubPlotter()
    nwp = object()
    field = _window_sum_field(plotter, nwp, torch.zeros(4, 2), [5], 1, (2, 2))
    np.testing.assert_allclose(field, (np.arange(4) + 5).reshape(2, 2))
    assert plotter.calls == [(5, nwp, "oat")]


def test_rank_stations_uses_box_coverage_and_order():
    obs, mask = _obs_and_mask()
    coords = torch.tensor([[0.0, 0.0], [1.0, 1.0], [50.0, 0.0]])
    ranked = _rank_stations(
        obs,
        mask,
        coords,
        eval_indices=[0, 1, 2],
        map_indices=[0, 1],
        ts_indices=[0, 1, 2, 3],
        var_idx=0,
        center=(0.0, 0.0),
        extent=10.0,
        use_sum=False,
    )
    # station 2 is outside the box; station 1 misses an hour of the ts window
    assert ranked == [(0, 18.0)]


def test_rank_stations_sum_mode_sorts_largest_first():
    obs = torch.zeros(3, 3, 1)
    obs[:, 0, 0] = 1.0
    obs[:, 1, 0] = 5.0
    obs[:, 2, 0] = 2.0
    mask = torch.ones(3, 3, 1, dtype=torch.bool)
    coords = torch.zeros(3, 2)
    ranked = _rank_stations(
        obs, mask, coords, [0, 1, 2], [0, 1], [0, 1, 2], 0, (0.0, 0.0), 1.0, True
    )
    assert ranked == [(1, 10.0), (2, 4.0), (0, 2.0)]


def test_daily_peak_means_uses_the_peak_hour_and_the_mask():
    times = pd.date_range("2024-07-12T00", periods=48, freq="h", tz="UTC")
    obs = torch.zeros(48, 2, 1)
    obs[14, :, 0] = torch.tensor([30.0, 34.0])
    obs[38, :, 0] = torch.tensor([36.0, 99.0])
    mask = torch.ones(48, 2, 1, dtype=torch.bool)
    mask[38, 1, 0] = False
    out = _daily_peak_means(obs, mask, times, list(range(48)), 0, [0, 1])
    assert [(t.day, v) for t, v in out] == [(12, 32.0), (13, 36.0)]
