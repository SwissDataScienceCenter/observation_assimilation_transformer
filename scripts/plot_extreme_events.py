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

import argparse
import gc
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D

from oat.data_module.setup import setup_data_and_transform
from oat.plotting_module.geo_overlays import _add_borders, build_domain_grid
from oat.predictor_module.checkpoint_utils import load_checkpoint, resolve_checkpoint

# The import of plot_timeseries also applies its rcParams block, so the fonts
# and line widths are the same as in the time-series figure.
from scripts.plot_timeseries import (
    _NWP_COLOR,
    _OAT_COLOR,
    FIG_WIDTH_IN,
    _load_units,
)
from scripts.plotter_utils import (
    load_plot_config,
    maybe_build_nwp_baseline,
    select_plotter_cls,
)

_MODEL = "oat_temp_precip"
_PROJECT_ROOT = Path(__file__).parents[1]
OUT_DIR = _PROJECT_ROOT / "data/figures"

_ROW_HEIGHT_IN = 4.8  # inches for one event row (maps, time series, titles)
_GRID_CELLS = 250  # map cells across the width, as in plot_assimilation_panel
_MAP_MARKER_SCALE = 0.5  # area factor for the station markers of overlay_stations
_MAP_MARKER_EDGE = 0.4  # edge width (points) of those markers
# Diverging colormap for differences, the same as plot_assimilation_panel.py.
# It is not RdBu_r, so it does not look like a temperature map.
_DIFF_CMAP = "PuOr"
_DIFF_PERCENTILE = 99.0  # colour limit of the Δ map, as in plot_ERA5_to_COSMO_grid
_OBS_DOT_SIZE = 8.0  # marker area (points^2) of the time-series obs dots
_CANDIDATE_PAD_DAYS = 7  # extra days on each side for the heat daily means
_VAR_LABELS = {"precipitation": "Precipitation", "temp": "Temperature"}

# Same proxy handles as plot_assimilation_panel.py, plus the plotted stations.
_STATION_HANDLES = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor="none",
        markeredgecolor="black",
        markersize=8,
        label="Training/evaluation station (assimilated)",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        linestyle="none",
        markerfacecolor="none",
        markeredgecolor="black",
        markersize=8,
        label="Test station (held out)",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor="white",
        markeredgecolor="grey",
        markersize=8,
        label="Missing observation",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        linestyle="none",
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=1.8,
        markersize=10,
        label="Test station in time series",
    ),
]


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


# The stations and times come from --list-candidates on the test split
# (2026-09-28). See the design spec for the selection rule.
EVENTS: list[ExtremeEvent] = [
    ExtremeEvent(
        dataset="cosmo_peakweather",
        variable="precipitation",
        title="(a) Maggia flood, 2024-06-29/30",
        map_start="2024-06-29:12",
        map_hours=24,
        ts_start="2024-06-29:06",
        ts_hours=48,
        stations=(185, 164),
        center=(2700000.0, 1135000.0),  # LV95, Ticino
        extent=75000.0,
    ),
    ExtremeEvent(
        dataset="era5_meteostat",
        variable="temp",
        title="(b) Heat wave, July 2024",
        map_start="2024-07-15:14",
        map_hours=1,
        ts_start="2024-07-12:00",
        ts_hours=168,
        stations=(1564, 1626),
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


def _station_mae(
    pred: np.ndarray, obs: np.ndarray, stations: Iterable[int]
) -> tuple[float, int]:
    """Return the MAE of *pred* against *obs* over *stations* and the count.

    Args:
        pred: [S] model window value for each station.
        obs: [S] obs window value. NaN for a station without a full window.

    Returns:
        ``(mae, n)``. A station with NaN is not used. The MAE is NaN if n is 0.
    """
    idx = [s for s in stations if np.isfinite(obs[s]) and np.isfinite(pred[s])]
    if not idx:
        return float("nan"), 0
    return float(np.mean(np.abs(pred[idx] - obs[idx]))), len(idx)


def _sum_unit(unit: str) -> str:
    """Return the unit of a sum over hours of a quantity with *unit*."""
    return unit.removesuffix("/h") if unit.endswith("/h") else f"{unit} h"


def _ts_slots(n_stations: int) -> list[slice]:
    """Return the gridspec row slices for the time-series axes of one row."""
    if n_stations == 1:
        return [slice(0, 2)]
    if n_stations == 2:
        return [slice(0, 1), slice(1, 2)]
    raise ValueError(f"An event needs 1 or 2 stations, not {n_stations}.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--list-candidates",
        action="store_true",
        help="Print the candidate held-out stations for each event and stop.",
    )
    p.add_argument("--n-samples", type=int, default=200, help="Time-series samples.")
    p.add_argument(
        "--map-n-samples",
        type=int,
        default=50,
        help="Samples for the map fields (the maps show only the mean).",
    )
    p.add_argument("--out", default=None, help="Output path (.pdf; a .png is added).")
    return p.parse_args(argv)


def _load_event(event: ExtremeEvent, logger: logging.Logger) -> tuple[Any, Any, Any]:
    """Load the checkpoint, the dataset and the test split for *event*."""
    checkpoint_path = resolve_checkpoint(event.dataset, _MODEL)
    predictor, cfg = load_checkpoint(None, checkpoint_path, map_location="cpu")
    predictor = predictor.eval()
    cfg.dataset.data_points_for_stats = 0
    dataset, _, _, _val_loader, test_loader, *_ = setup_data_and_transform(cfg, logger)
    # load_from_checkpoint does not restore the dataset-derived projection.
    # Without it, ERA5 checkpoints crash with a geo-width shape mismatch.
    if hasattr(predictor, "set_index_projection"):
        predictor.set_index_projection(dataset)
    return predictor, dataset, test_loader.dataset


def _station_name(dataset: Any, s: int) -> str:
    """Return the station ID of index *s* (``dataset.stations``), if it exists."""
    ids = getattr(dataset, "stations", None)
    return str(ids[s]) if ids is not None else f"station {s}"


def _print_candidates(
    event: ExtremeEvent, predictor: Any, dataset: Any, test_ds: Any
) -> None:
    """Print the ranked candidate stations (and, for heat, the daily means)."""
    eval_indices = [int(i) for i in test_ds.eval_indices]
    map_indices = _window_time_indices(dataset.times, event.map_start, event.map_hours)
    ts_indices = _window_time_indices(dataset.times, event.ts_start, event.ts_hours)
    var_idx = predictor.var_to_indx[event.variable]
    use_sum = event.map_hours > 1
    ranked = _rank_stations(
        dataset.obs_tensor,
        dataset.obs_mask_tensor,
        dataset.station_coords,
        eval_indices,
        map_indices,
        ts_indices,
        var_idx,
        event.center,
        event.extent,
        use_sum,
    )
    what = f"obs {event.map_hours} h sum" if use_sum else "obs maximum"
    print(
        f"\n{event.title} [{event.dataset}]: {len(ranked)} held-out stations in "
        f"the map box with all obs in the time-series window, ranked by {what}:"
    )
    xy = np.asarray(dataset.station_coords, dtype=float)
    for s, value in ranked[:15]:
        print(
            f"  [{s:>5d}] {_station_name(dataset, s):<24s} "
            f"x={xy[s, 0]:<11.6g} y={xy[s, 1]:<11.6g} {value:8.2f}"
        )
    if use_sum or not ranked:
        return
    pad = pd.Timedelta(days=_CANDIDATE_PAD_DAYS)
    wide_start = (_parse_time(event.ts_start) - pad).strftime("%Y-%m-%d:%H")
    wide_hours = event.ts_hours + 2 * _CANDIDATE_PAD_DAYS * 24
    wide = _window_time_indices(dataset.times, wide_start, wide_hours)
    print(f"  Daily mean of these stations' obs at 14 UTC (window ± {pad.days} d):")
    means = _daily_peak_means(
        dataset.obs_tensor,
        dataset.obs_mask_tensor,
        dataset.times,
        wide,
        var_idx,
        [s for s, _ in ranked],
    )
    for ts, value in means:
        print(f"    {ts:%Y-%m-%d}  {value:6.2f}")


def _draw_event_row(
    subfig: Any,
    event: ExtremeEvent,
    predictor: Any,
    dataset: Any,
    test_ds: Any,
    args: argparse.Namespace,
    logger: logging.Logger,
    with_legend: bool,
) -> None:
    """Draw the NWP map, the OAT map and the time series of *event*.

    If *with_legend* is True, add the station legend below the maps and the
    time-series legend below the time series.
    """
    obs_indices = [int(i) for i in test_ds.observed_indices]
    eval_indices = [int(i) for i in test_ds.eval_indices]
    map_indices = _window_time_indices(dataset.times, event.map_start, event.map_hours)
    ts_indices = _window_time_indices(dataset.times, event.ts_start, event.ts_hours)
    _validate_event(
        event,
        eval_indices,
        [int(t) for t in test_ds._time_indices],
        map_indices,
        ts_indices,
    )

    var_idx = predictor.var_to_indx[event.variable]
    plot_config = load_plot_config(dataset)
    var_cfg = (
        plot_config[event.variable]
        if (plot_config is not None and event.variable in plot_config)
        else {}
    )
    cmap = var_cfg.get("cmap", "viridis")
    unit = _load_units(event.dataset).get(event.variable, "")
    var_label = _VAR_LABELS.get(event.variable, event.variable.capitalize())
    if event.map_hours > 1:
        map_label = f"{event.map_hours} h {var_label.lower()}"
        map_unit = _sum_unit(unit) if unit else ""
    else:
        map_label, map_unit = var_label, unit

    nwp_predictor = maybe_build_nwp_baseline(dataset, logger, required=True)
    plotter_cls = select_plotter_cls(dataset)

    def _plotter(n_samples: int) -> Any:
        return plotter_cls(
            predictor=predictor,
            dataset=dataset,
            obs_station_indices=obs_indices,
            eval_indices=eval_indices,
            n_samples=n_samples,
            device="cpu",
            plot_config=plot_config,
        )

    map_plotter = _plotter(args.map_n_samples)
    ts_plotter = _plotter(args.n_samples)

    # ---- Maps ----
    xs, ys = build_domain_grid(dataset, center=event.center, extent=event.extent)
    xs, ys = build_domain_grid(
        dataset,
        resolution=(xs[-1] - xs[0]) / _GRID_CELLS,
        center=event.center,
        extent=event.extent,
    )
    x_grid, y_grid = np.meshgrid(xs, ys)
    grid_coords = torch.tensor(
        np.stack([x_grid.ravel(), y_grid.ravel()], axis=1), dtype=torch.float32
    )
    logger.info(
        "%s: grid %dx%d, %d map hours", event.title, len(xs), len(ys), len(map_indices)
    )
    fields = {
        name: _window_sum_field(
            map_plotter, p, grid_coords, map_indices, var_idx, y_grid.shape
        )
        for name, p in (("NWP interpolation", nwp_predictor), ("OAT", None))
    }
    vmin = float(np.nanmin([np.nanmin(f) for f in fields.values()]))
    vmax = float(np.nanmax([np.nanmax(f) for f in fields.values()]))
    diff = fields["OAT"] - fields["NWP interpolation"]

    # Window values at the drawn stations: obs (NaN without a full window),
    # NWP and OAT. They give the obs - NWP markers and the test-station MAE.
    coords = np.asarray(dataset.station_coords, dtype=float)
    drawn = obs_indices + eval_indices
    obs_win = _station_window_obs(
        dataset.obs_tensor, dataset.obs_mask_tensor, map_indices, var_idx
    )
    drawn_coords = torch.tensor(coords[drawn], dtype=torch.float32)
    pred_win: dict[str, np.ndarray] = {}
    for name, p in (("NWP interpolation", nwp_predictor), ("OAT", None)):
        full = np.full(len(coords), np.nan)
        full[drawn] = _window_sum_field(
            map_plotter, p, drawn_coords, map_indices, var_idx, (len(drawn),)
        )
        pred_win[name] = full
    innovation = obs_win - pred_win["NWP interpolation"]
    color_by_diff = {
        s: float(innovation[s]) for s in drawn if np.isfinite(innovation[s])
    }
    # The scale covers the field and the markers on the map, or the markers
    # saturate. As in plot_ERA5_to_COSMO_grid.py, the limit is a percentile of
    # |Δ|, so a small number of outliers cannot make the map white.
    on_map = [
        s
        for s in color_by_diff
        if xs[0] <= coords[s, 0] <= xs[-1] and ys[0] <= coords[s, 1] <= ys[-1]
    ]
    all_diff = np.concatenate([diff.ravel(), [color_by_diff[s] for s in on_map]])
    diff_max = float(np.nanpercentile(np.abs(all_diff), _DIFF_PERCENTILE)) or 1.0

    # MAE only at the held-out stations that are in the map box.
    in_box = [
        s
        for s in eval_indices
        if xs[0] <= coords[s, 0] <= xs[-1] and ys[0] <= coords[s, 1] <= ys[-1]
    ]
    titles = {}
    for name in fields:
        mae, n = _station_mae(pred_win[name], obs_win, in_box)
        logger.info("%s: %s MAE %.3f at %d test stations", event.title, name, mae, n)
        titles[name] = f"{name}\nMAE {mae:.1f} {map_unit}".rstrip()

    color_by: dict[int, float] | None = None
    drop_xy = np.zeros((0, 2))
    context_xy = np.zeros((0, 2))
    if event.map_hours > 1:
        color_by = {s: float(v) for s, v in enumerate(obs_win) if np.isfinite(v)}
        obs_set = set(obs_indices)
        no_sum = [s for s in drawn if not np.isfinite(obs_win[s])]
        drop_xy = coords[no_sum].reshape(-1, 2)
        context_xy = coords[[s for s in no_sum if s in obs_set]].reshape(-1, 2)

    labels = [str(i + 1) for i in range(len(event.stations))]
    # Nested subfigures give the time series one shared y-label that does not
    # overlap the colorbar of the maps.
    map_fig, ts_fig = subfig.subfigures(1, 2, width_ratios=[3.0, 1.5])
    ax_maps = list(map_fig.subplots(1, 3))
    panels = [
        (titles["NWP interpolation"], fields["NWP interpolation"], cmap, vmin, vmax),
        (titles["OAT"], fields["OAT"], cmap, vmin, vmax),
        (
            "\u0394 = OAT \u2212 NWP\nmarkers: obs \u2212 NWP",
            diff,
            _DIFF_CMAP,
            -diff_max,
            diff_max,
        ),
    ]
    meshes = []
    for ax, (title, field, p_cmap, p_vmin, p_vmax) in zip(ax_maps, panels, strict=True):
        mesh = map_plotter.draw_field(
            ax, field, xs, ys, cmap=p_cmap, vmin=p_vmin, vmax=p_vmax
        )
        # Rasterize the dense field so the PDF stays small.
        mesh.set_rasterized(True)
        meshes.append(mesh)
        # The difference panel colours the markers by obs - NWP.
        is_diff = ax is ax_maps[2]
        map_plotter.overlay_stations(
            ax,
            map_indices[0],
            var_idx,
            mesh=mesh,
            color_by=color_by_diff if is_diff else color_by,
        )
        _restyle_window_missing(ax, drop_xy, context_xy)
        for coll in list(ax.collections):
            if coll.get_label() == "eval missing":
                coll.remove()
            elif isinstance(coll, PathCollection):
                # Smaller markers with thin edges keep dense networks legible.
                coll.set_sizes(coll.get_sizes() * _MAP_MARKER_SCALE)
                coll.set_linewidths(_MAP_MARKER_EDGE)
        _mark_stations(ax, coords[list(event.stations)], labels)
        ax.set_title(title, fontsize=13)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    map_fig.colorbar(
        meshes[0],
        ax=ax_maps[:2],
        label=f"{map_label} [{map_unit}]" if map_unit else map_label,
        shrink=0.85,
    )
    # extend="both": the limits are percentiles, and some data is outside them.
    map_fig.colorbar(
        meshes[2],
        ax=ax_maps[2],
        label=f"Δ [{map_unit}]" if map_unit else "Δ",
        shrink=0.85,
        extend="both",
    )
    try:
        _add_borders(ax_maps, xs, ys)
    except Exception as exc:
        logger.warning("Could not draw borders: %s", exc)

    # ---- Time series ----
    gs_ts = ts_fig.add_gridspec(2, 1)
    ax_ts = [ts_fig.add_subplot(gs_ts[sl, 0]) for sl in _ts_slots(len(event.stations))]
    for ax in ax_ts[1:]:
        ax.sharex(ax_ts[0])
    for ax, s, label in zip(ax_ts, event.stations, labels, strict=True):
        ts_plotter.plot_timeseries_at_station(
            ax=ax,
            station_idx=s,
            var=event.variable,
            time_indices=ts_indices,
            quantile_lo=0.05,
            quantile_hi=0.95,
            predictors={"NWP interpolation": nwp_predictor},
            stage="oat",
            label="OAT",
            color=_OAT_COLOR,
        )
        # The plotter draws the obs dots at the default size. Make them smaller.
        for coll in ax.collections:
            if coll.get_label() == "observation":
                coll.set_sizes([_OBS_DOT_SIZE])
        # plot_timeseries draws the NWP line without a colour; recolour it
        # the same as plot_timeseries.py.
        for line in ax.get_lines():
            if line.get_label() == "NWP interpolation":
                line.set_color(_NWP_COLOR)
                line.set_linestyle("--")
        # Show the map time. For a single map hour, the band is 1 h wide.
        ax.axvspan(
            dataset.times[map_indices[0]],
            dataset.times[map_indices[-1]] + pd.Timedelta(hours=1),
            color="0.85",
            zorder=0,
            label="Map window",
        )
        ax.set_title(f"{label}: {_station_name(dataset, s)}", loc="left", fontsize=13)
        ax.grid(alpha=0.3)
    for ax in ax_ts[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    ax_ts[-1].set_xlabel("Time (UTC)")
    ax_ts[-1].tick_params(axis="x", rotation=30)
    ts_fig.supylabel(
        f"{var_label} [{unit}]" if unit else var_label,
        fontsize=plt.rcParams["axes.labelsize"],
    )

    subfig.suptitle(event.title, x=0.0, ha="left")
    if with_legend:
        map_fig.legend(
            handles=_STATION_HANDLES,
            loc="outside lower center",
            ncol=2,
            frameon=False,
            fontsize=11,
        )
        # Below the time series, so the legend cannot cover the data.
        ts_fig.legend(
            *ax_ts[0].get_legend_handles_labels(),
            loc="outside lower center",
            ncol=3,
            frameon=False,
            fontsize=11,
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)
    args = _parse_args()

    if args.list_candidates:
        for event in EVENTS:
            predictor, dataset, test_ds = _load_event(event, logger)
            _print_candidates(event, predictor, dataset, test_ds)
            del predictor, dataset, test_ds
            gc.collect()
        return

    n_rows = len(EVENTS)
    fig = plt.figure(
        figsize=(FIG_WIDTH_IN, n_rows * _ROW_HEIGHT_IN), layout="constrained"
    )
    # One subfigure per event. Only the top row has legends.
    subfigs = fig.subfigures(n_rows, 1, squeeze=False)[:, 0]
    for row, (subfig, event) in enumerate(zip(subfigs, EVENTS, strict=True)):
        predictor, dataset, test_ds = _load_event(event, logger)
        _draw_event_row(
            subfig,
            event,
            predictor,
            dataset,
            test_ds,
            args,
            logger,
            with_legend=row == 0,
        )
        # Free the dataset before the next one loads.
        del predictor, dataset, test_ds
        gc.collect()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(args.out) if args.out is not None else OUT_DIR / "extreme_events.pdf"
    )
    fig.savefig(str(out_path.with_suffix(".pdf")), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_path.with_suffix(".png")), dpi=150, bbox_inches="tight")
    logger.info("Saved %s (.pdf and .png)", out_path.with_suffix(""))


if __name__ == "__main__":
    main()
