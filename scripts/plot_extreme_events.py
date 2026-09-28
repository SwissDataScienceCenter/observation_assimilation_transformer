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
from typing import Any

import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import torch
from matplotlib.collections import PathCollection

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


def _station_window_obs(
    obs: torch.Tensor, mask: torch.Tensor, time_indices: list[int], var_idx: int
) -> np.ndarray:
    """Return the sum of the obs over *time_indices* for each station.

    Args:
        obs: [T, S, V] observations. A masked entry can hold any value (NaN too).
        mask: [T, S, V] observation mask (True = observed).

    Returns:
        [S] float64 sums. NaN for a station that misses one or more hours.
    """
    o = obs[time_indices][:, :, var_idx].double()
    m = mask[time_indices][:, :, var_idx].bool()
    total = torch.where(m, o, torch.zeros_like(o)).sum(dim=0)
    total[~m.all(dim=0)] = float("nan")
    return total.numpy()


def _window_sum_field(
    plotter: Any,
    predictor: Any,
    grid_coords: torch.Tensor,
    time_indices: list[int],
    var_idx: int,
    shape: tuple[int, int],
) -> np.ndarray:
    """Return the sum over *time_indices* of the per-hour sample-mean field.

    The sum of the hourly means equals the mean of the window sums. For one
    hour, the result is the mean field of that hour. ``predictor=None`` uses
    the main model of *plotter*.
    """
    total = np.zeros(shape)
    for t in time_indices:
        samples = plotter.predict_samples(
            grid_coords, t, predictor=predictor, stage="oat"
        )  # [N_q, n_samples, n_vars]
        total += (
            samples[:, :, var_idx].mean(dim=1).detach().cpu().numpy().reshape(shape)
        )
    return total


def _rank_stations(
    obs: torch.Tensor,
    mask: torch.Tensor,
    station_coords: Any,
    eval_indices: Iterable[int],
    map_indices: list[int],
    ts_indices: list[int],
    var_idx: int,
    center: tuple[float, float],
    extent: float,
    use_sum: bool,
) -> list[tuple[int, float]]:
    """Rank the held-out stations in the map box by their extreme obs value.

    A station is a candidate if it is inside ``center ± extent`` and has all
    obs of the time-series window. The value is the obs sum over the map
    window (*use_sum*) or the obs maximum over the time-series window.

    Returns:
        ``(station, value)`` pairs, largest value first.
    """
    full = mask[ts_indices][:, :, var_idx].bool().all(dim=0).numpy()
    if use_sum:
        values = _station_window_obs(obs, mask, map_indices, var_idx)
    else:
        values = obs[ts_indices][:, :, var_idx].double().amax(dim=0).numpy()
    xy = np.asarray(station_coords, dtype=float)
    ranked = []
    for s in (int(i) for i in eval_indices):
        in_box = (
            abs(xy[s, 0] - center[0]) <= extent and abs(xy[s, 1] - center[1]) <= extent
        )
        if in_box and full[s] and np.isfinite(values[s]):
            ranked.append((s, float(values[s])))
    return sorted(ranked, key=lambda r: r[1], reverse=True)


def _daily_peak_means(
    obs: torch.Tensor,
    mask: torch.Tensor,
    times: pd.DatetimeIndex,
    time_indices: list[int],
    var_idx: int,
    stations: list[int],
    hour: int = 14,
) -> list[tuple[pd.Timestamp, float]]:
    """Return the mean obs of *stations* at *hour* UTC for each day.

    The mean uses only observed entries. A day with no obs is left out.
    """
    out = []
    for t in time_indices:
        if times[t].hour != hour:
            continue
        m = mask[t, stations, var_idx].bool()
        if m.any():
            out.append((times[t], float(obs[t, stations, var_idx][m].double().mean())))
    return out


def _restyle_window_missing(
    ax: Any, drop_xy: np.ndarray, context_xy: np.ndarray
) -> None:
    """Remove the markers at *drop_xy*, then show *context_xy* as missing.

    ``overlay_stations`` takes the missing state from one hour. For a window
    sum, a station that misses one hour of the window has no valid value. This
    removes its marker from each station scatter on *ax*. Context stations get
    the grey "missing" circle of ``overlay_stations`` again. Held-out stations
    get no marker, because the figure removes the "eval missing" squares.
    """
    if len(drop_xy):
        for coll in ax.collections:
            if not isinstance(coll, PathCollection):
                continue
            off = np.asarray(coll.get_offsets())
            if not len(off):
                continue
            hit = (
                np.isclose(off[:, None, :], drop_xy[None, :, :], rtol=0.0, atol=1e-6)
                .all(axis=2)
                .any(axis=1)
            )
            if not hit.any():
                continue
            face = coll.get_facecolors()
            coll.set_offsets(off[~hit])
            if len(face) == len(off):
                coll.set_facecolors(face[~hit])
    if len(context_xy):
        # Same style as the "missing" markers of OATPlotter._overlay_stations.
        ax.scatter(
            context_xy[:, 0],
            context_xy[:, 1],
            marker="o",
            facecolors="white",
            edgecolors="grey",
            s=10,
            zorder=5,
            label="missing",
        )


def _mark_stations(ax: Any, xy: np.ndarray, labels: list[str]) -> None:
    """Draw a thick square and a number label on each plotted test station."""
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        marker="s",
        s=60,
        facecolors="none",
        edgecolors="black",
        linewidths=1.8,
        zorder=6,
    )
    for (x, y), label in zip(xy, labels, strict=True):
        ax.annotate(
            label,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=13,
            fontweight="bold",
            zorder=7,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
        )


def _ts_slots(n_stations: int) -> list[slice]:
    """Return the gridspec row slices for the time-series axes of one row."""
    if n_stations == 1:
        return [slice(0, 2)]
    if n_stations == 2:
        return [slice(0, 1), slice(1, 2)]
    raise ValueError(f"An event needs 1 or 2 stations, not {n_stations}.")
