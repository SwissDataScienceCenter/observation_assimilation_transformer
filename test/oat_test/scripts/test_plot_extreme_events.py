import dataclasses

import pandas as pd
import pytest

from scripts.plot_extreme_events import (
    EVENTS,
    _parse_time,
    _validate_event,
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
