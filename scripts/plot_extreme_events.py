"""Paper figure: OAT during two extreme events in the test period.

The figure has one row for each event. Each row shows the NWP interpolation
map, the OAT mean map, and the time series at 1-2 held-out test stations:
  (a) Switzerland (COSMO/PeakWeather): Maggia flood, 24 h precipitation sum.
  (b) Europe (ERA5/Meteostat): heat wave in July 2024, temperature at 14 UTC.

The events are fixed in ``EVENTS``. The script loads one dataset at a time.

Usage
-----
List the candidate held-out stations for each event (no figure):
    python -m scripts.plot_extreme_events --list-candidates

Make the figure:
    python -m scripts.plot_extreme_events --n-samples 200 --map-n-samples 50
"""

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

_MODEL = "oat_temp_precip"


@dataclass(frozen=True)
class ExtremeEvent:
    """One figure row: an extreme event in the test period of one dataset."""

    dataset: str  # "cosmo_peakweather" | "era5_meteostat"
    variable: str  # observed-variable name of the dataset
    title: str  # row title
    map_start: str  # "YYYY-MM-DD:HH" UTC; start of the sum window for precip
    map_hours: int  # >1: sum over the window; 1: one hour
    ts_start: str  # "YYYY-MM-DD:HH" UTC
    ts_hours: int
    stations: tuple[int, ...]  # held-out test station indices
    center: tuple[float, float]  # map centre, in the dataset frame
    extent: float  # map half-width, dataset units (see build_domain_grid)


# The Europe stations and the peak hour are provisional. Task 5 of the plan
# replaces them with the result of --list-candidates.
EVENTS: list[ExtremeEvent] = [
    ExtremeEvent(
        dataset="cosmo_peakweather",
        variable="precipitation",
        title="(a) Maggia flood, 2024-06-29/30",
        map_start="2024-06-29:12",
        map_hours=24,
        ts_start="2024-06-28:12",
        ts_hours=72,
        stations=(185, 164),
        center=(2700000.0, 1135000.0),  # LV95, Ticino
        extent=75000.0,
    ),
    ExtremeEvent(
        dataset="era5_meteostat",
        variable="temp",
        title="(b) Heat wave, July 2024",
        map_start="2024-07-16:14",
        map_hours=1,
        ts_start="2024-07-12:00",
        ts_hours=168,
        stations=(1572, 1563),
        center=(18.0, 42.0),  # lon, lat: Italy and the Balkans
        extent=12.0,
    ),
]


def _parse_time(time_str: str) -> pd.Timestamp:
    """Parse ``YYYY-MM-DD:HH`` (or ``YYYY-MM-DD``) as a UTC timestamp."""
    return pd.Timestamp(time_str.replace(":", " "), tz="UTC")


def _window_time_indices(times: pd.DatetimeIndex, start: str, hours: int) -> list[int]:
    """Return the dataset indices of *hours* hourly steps from *start*.

    Raise if one step is not in *times* (a missing date or the dataset end).
    A nearest-time fallback would move the window and make the title wrong.
    """
    wanted = pd.date_range(_parse_time(start), periods=hours, freq="h")
    missing = wanted.difference(times)
    if len(missing):
        raise ValueError(
            f"{len(missing)} of {hours} hourly steps from {wanted[0]} are not in "
            f"the dataset times (first: {missing[0]})."
        )
    return [int(times.get_loc(t)) for t in wanted]


def _validate_event(
    event: ExtremeEvent,
    eval_indices: Iterable[int],
    test_time_indices: Iterable[int],
    map_indices: list[int],
    ts_indices: list[int],
) -> None:
    """Raise if a station is not held out or a time is not in the test period."""
    eval_set = {int(s) for s in eval_indices}
    not_held_out = [s for s in event.stations if s not in eval_set]
    if not_held_out:
        raise ValueError(
            f"{event.title}: stations are not held-out test stations: "
            f"{not_held_out}"
        )
    test_set = {int(t) for t in test_time_indices}
    outside = sorted((set(map_indices) | set(ts_indices)) - test_set)
    if outside:
        raise ValueError(
            f"{event.title}: {len(outside)} time steps are outside the test "
            f"period (first dataset index: {outside[0]})."
        )
