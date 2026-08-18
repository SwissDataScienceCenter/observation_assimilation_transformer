"""Covariate computation, binning and stats for the error-stratification plot."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

NDArray = np.ndarray[Any, np.dtype[Any]]

SEASONS = ["DJF", "MAM", "JJA", "SON"]
_MONTH_TO_SEASON = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}


def detect_lonlat(coords: NDArray) -> tuple[NDArray, NDArray]:
    """Return (lon, lat) for station coords, auto-detecting the CRS.

    Swiss LV95 (EPSG:2056, easting/northing in metres) is detected when the
    first column exceeds 1e5 and reprojected to WGS84. Otherwise the coords are
    assumed to already be [lon, lat] (era5_meteostat).
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.size == 0:
        return coords[:, 0], coords[:, 1]
    if np.abs(coords[:, 0]).max() > 1e5:
        from pyproj import Transformer

        tf = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
        lon, lat = tf.transform(coords[:, 0], coords[:, 1])
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    return coords[:, 0], coords[:, 1]


_EARTH_RADIUS_KM = 6371.0088  # mean Earth radius (IUGG)


def _haversine_km(
    lon1: NDArray, lat1: NDArray, lon2: NDArray, lat2: NDArray
) -> NDArray:
    """Great-circle distance (km) between broadcastable lon/lat arrays (deg)."""
    lon1, lat1, lon2, lat2 = (np.radians(a) for a in (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def nearest_obs_distance(eval_coords: NDArray, obs_coords: NDArray) -> NDArray:
    """Great-circle distance (km) from each eval station to its nearest obs.

    Both inputs are ``[*, 2]`` arrays of WGS84 ``[lon, lat]`` in degrees, so the
    returned distances are true geodesic kilometres regardless of the source
    CRS. Returns all-NaN when there are no observed stations (e.g. the
    uninformed baseline), where the covariate is undefined; such rows are then
    excluded from the nearest-obs-distance stratification.
    """
    eval_coords = np.asarray(eval_coords, dtype=np.float64)
    obs_coords = np.asarray(obs_coords, dtype=np.float64)
    if obs_coords.shape[0] == 0:
        return np.full(eval_coords.shape[0], np.nan, dtype=np.float64)
    dists = _haversine_km(
        eval_coords[:, 0][:, None],
        eval_coords[:, 1][:, None],
        obs_coords[:, 0][None, :],
        obs_coords[:, 1][None, :],
    )
    return np.asarray(dists.min(axis=1))


def season_labels(times: Any) -> NDArray:
    """Map a sequence of timestamps to DJF/MAM/JJA/SON labels."""
    months = pd.DatetimeIndex(times).month
    return np.array([_MONTH_TO_SEASON[int(m)] for m in months], dtype=object)


_ELEVATION_NAMES = ("dem", "elevation")


def _station_elevation(dataset: Any) -> NDArray:
    """Read per-station elevation from the dataset's geo-bias field.

    Field datasets (post-#116) hold per-station statics in ``fields``, not on
    a ``station_geo`` attribute -- pull the elevation channel straight from
    the geo-bias field of one sample (its values don't vary across time).
    """
    geo_name = dataset.roles()["geo_bias"]
    geo_field = dataset.field_config.fields[geo_name]
    try:
        channel = next(i for i, v in enumerate(geo_field.out) if v in _ELEVATION_NAMES)
    except StopIteration as exc:
        raise ValueError(
            f"No elevation-like channel (one of {_ELEVATION_NAMES}) found in "
            f"geo-bias field {geo_name!r} outputs {list(geo_field.out)!r}"
        ) from exc
    sample = dataset[0]["fields"][geo_name]
    return np.asarray(sample.values[:, channel].cpu().numpy())


def collect_station_covariates(
    dataset: Any, eval_indices: list[int], obs_indices: list[int]
) -> dict[str, NDArray]:
    """Compute static per-eval-station covariates for the long dump."""
    coords = dataset.station_coords.cpu().numpy()
    eval_coords = coords[eval_indices]
    obs_coords = coords[obs_indices]
    elevation = _station_elevation(dataset)
    lon, lat = detect_lonlat(eval_coords)
    # Reproject both sides to WGS84 so nearest-obs distance is geodesic km.
    obs_lonlat = (
        np.column_stack(detect_lonlat(obs_coords))
        if obs_coords.shape[0]
        else obs_coords
    )
    return {
        "station_idx": np.asarray(eval_indices),
        "station_x": eval_coords[:, 0],
        "station_y": eval_coords[:, 1],
        "lon": lon,
        "lat": lat,
        "elevation": elevation[eval_indices],
        "nearest_obs_dist": nearest_obs_distance(
            np.column_stack([lon, lat]), obs_lonlat
        ),
    }


def build_long_dataframe(
    *,
    crps: NDArray,
    mae: NDArray,
    times: Any,
    covariates: dict[str, NDArray],
    var_names: list[str],
) -> pd.DataFrame:
    """Assemble a per-(station, timestep) long DataFrame.

    Args:
        crps: [T, S, V] per-(time, station, var) CRPS (NaN where no obs).
        mae: [T, S, V] per-(time, station, var) MAE (NaN where no obs).
        times: length-T sequence of timestamps.
        covariates: output of ``collect_station_covariates`` (length-S arrays).
        var_names: length-V list of variable names.

    Rows are ordered time-major: (t0,s0), (t0,s1), ..., (t1,s0), ...
    """
    T, S, _ = crps.shape
    seasons = season_labels(times)  # [T]
    data: dict[str, NDArray] = {
        "station_idx": np.tile(covariates["station_idx"], T),
        "time": np.repeat(np.asarray(pd.DatetimeIndex(times).values), S),
        "season": np.repeat(seasons, S),
        "station_x": np.tile(covariates["station_x"], T),
        "station_y": np.tile(covariates["station_y"], T),
        "lon": np.tile(covariates["lon"], T),
        "lat": np.tile(covariates["lat"], T),
        "elevation": np.tile(covariates["elevation"], T),
        "nearest_obs_dist": np.tile(covariates["nearest_obs_dist"], T),
    }
    for vi, var in enumerate(var_names):
        data[f"crps_{var}"] = crps[:, :, vi].reshape(-1)
        data[f"mae_{var}"] = mae[:, :, vi].reshape(-1)
    return pd.DataFrame(data)


def _assign_fixed_width_bins(values: NDArray, nbins: int) -> tuple[NDArray, NDArray]:
    """Assign fixed-width bin indices; return (bin_idx, edges)."""
    finite = values[np.isfinite(values)]
    lo, hi = float(finite.min()), float(finite.max())
    if lo == hi:
        edges = np.array([lo, lo + 1.0])
    else:
        edges = np.linspace(lo, hi, nbins + 1)
    # digitize on interior edges; clip to valid bin range
    idx = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
    return idx, edges


def _assign_integer_bins(values: NDArray, nbins: int) -> tuple[NDArray, NDArray]:
    """Fixed-width bins snapped to whole-number edges; return (bin_idx, edges).

    Edges span ``floor(min)..ceil(max)`` with an integer width chosen to land
    near ``nbins`` bins, so lon/lat bins fall on full degrees.
    """
    finite = values[np.isfinite(values)]
    lo = float(np.floor(finite.min()))
    hi = float(np.ceil(finite.max()))
    width = max(1, round((hi - lo) / max(1, nbins)))
    edges = np.arange(lo, hi + width, width, dtype=float)
    if edges.size < 2:
        edges = np.array([lo, lo + width], dtype=float)
    idx = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
    return idx, edges


def _bin_label(edges: NDArray, i: int) -> str:
    lo, hi = edges[i], edges[i + 1]
    width = abs(hi - lo)
    # choose decimals so adjacent edges are distinguishable
    if width >= 100:
        fmt = "{:.0f}-{:.0f}"
    elif width >= 1:
        fmt = "{:.1f}-{:.1f}"
    else:
        fmt = "{:.2f}-{:.2f}"
    return fmt.format(lo, hi)


def _bin_keys(
    df: pd.DataFrame, stratifier: str, nbins: int
) -> tuple[NDArray, list[Any], dict[Any, str], dict[Any, int]]:
    """Return (keys, present, label_for, order_for) for a stratifier column.

    keys: per-row bin key aligned to df rows (season string, or int bin index).
    present: ordered list of bin keys that occur.
    label_for: bin key -> display label.
    order_for: bin key -> integer sort order.
    """
    keys: NDArray
    present: list[Any]
    label_for: dict[Any, str]
    order_for: dict[Any, int]

    if stratifier == "season":
        keys = df["season"].to_numpy()
        present = [s for s in SEASONS if s in set(keys)]
        label_for = {s: s for s in present}
        order_for = {s: i for i, s in enumerate(present)}
        return keys, present, label_for, order_for

    values = df[stratifier].to_numpy(dtype=float)
    finite = np.isfinite(values)
    # Guard against all-NaN stratifier
    if finite.sum() == 0:
        return np.full(len(df), -1, dtype=int), [], {}, {}
    if stratifier in _CUSTOM_BINS:
        interior = np.asarray(_CUSTOM_BINS[stratifier]["edges"], dtype=float)
        labels = _CUSTOM_BINS[stratifier]["labels"]
        keys = np.digitize(values, interior).astype(int)
        keys[~finite] = -1
        present = sorted({int(i) for i in keys[finite]})
        label_for = {i: labels[i] for i in present}
        order_for = {i: i for i in present}
        return keys, present, label_for, order_for
    if stratifier in _INTEGER_BIN_STRATIFIERS:
        idx, edges = _assign_integer_bins(values, nbins)
        keys = idx.astype(int)
        keys[~finite] = -1
        present = sorted({int(i) for i in keys[finite]})
        label_for = {i: f"{int(edges[i])}-{int(edges[i + 1])}" for i in present}
        order_for = {i: i for i in present}
        return keys, present, label_for, order_for
    idx, edges = _assign_fixed_width_bins(values, nbins)
    keys = idx.astype(int)
    # NaN covariate values (e.g. a station outside the topo extent) must NOT be
    # folded into the last bin by np.digitize's clip — assign them a sentinel
    # key so they are excluded from every real bin.
    keys[~finite] = -1
    present = sorted({int(i) for i in keys[finite]})
    label_for = {i: _bin_label(edges, i) for i in present}
    order_for = {i: i for i in present}
    return keys, present, label_for, order_for


STRATIFIERS = ["season", "elevation", "nearest_obs_dist", "lon", "lat"]

# Stratifiers with fixed, hand-chosen bins. ``edges`` are the interior cut
# points fed to np.digitize; ``labels`` (one per resulting bin) describe the
# open first/last intervals explicitly.
_CUSTOM_BINS: dict[str, dict[str, Any]] = {
    "elevation": {
        "edges": [750.0, 1250.0, 2000.0],
        "labels": ["<750", "750-1250", "1250-2000", ">2000"],
    },
}

# Stratifiers whose bin edges are snapped to whole numbers (degrees).
_INTEGER_BIN_STRATIFIERS = {"lon", "lat"}

_STRATIFIER_LABEL = {
    "season": "Season",
    "elevation": "Elevation [m]",
    "nearest_obs_dist": "Dist. to nearest obs. [km]",
    "lon": "Longitude [°E]",
    "lat": "Latitude [°N]",
}

# Okabe & Ito (2008) colourblind-safe categorical palette, in assignment order.
# Colour follows the model identity, never its rank; a 5th model is not supported
# (fold into "other" or use small multiples instead of cycling hues).
_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

# Registry/group keys -> display names for legend and axis text.
_MODEL_DISPLAY = {
    "oat": "OAT",
    "uninformed": "Uninformed",
    "uninformed_oat": "Uninformed",
    "gp": "GP",
    "deep_gp": "Deep GP",
    "ootg": "OOTG",
    "interpolator": "NWP interp.",
}


# Variable keys differ by dataset (cosmo_peakweather: temperature/precipitation;
# era5_meteostat: temp/prcp) -> shared display names.
_VAR_DISPLAY = {
    "temp": "Temperature",
    "temperature": "Temperature",
    "prcp": "Precipitation",
    "precip": "Precipitation",
    "precipitation": "Precipitation",
    "wspd": "Wind speed",
    "wind_speed": "Wind speed",
}


def _display_model(key: str) -> str:
    return _MODEL_DISPLAY.get(key, key)


def _display_var(var: str) -> str:
    return _VAR_DISPLAY.get(var.lower(), var.replace("_", " ").capitalize())


def _is_precip(var: str) -> bool:
    """Heavy-tailed precipitation variable (``precipitation``/``precip``/``prcp``)."""
    v = var.lower()
    return "precip" in v or "prcp" in v


def _symlog_linthresh(values: NDArray) -> float:
    """Pick a symlog linear-threshold for a heavy-tailed, non-negative metric.

    CRPS is non-negative in theory; values at/below zero are numerical noise.
    The threshold is set to swallow that noise — it is at least the magnitude of
    the most-negative value and the lower quartile of the positive values — so
    the near-zero bulk collapses into a thin linear band around zero while the
    meaningful upper spread of the boxes is resolved on the log region above it.
    """
    finite = values[np.isfinite(values)]
    pos = finite[finite > 0]
    if pos.size == 0:
        return 1e-6
    neg_mag = float(-min(finite.min(initial=0.0), 0.0))
    return max(neg_mag, float(np.percentile(pos, 25)), 1e-12)


def make_stratification_figure(
    df: pd.DataFrame,
    var_names: list[str],
    metric: str,
    model_order: list[str],
    nbins: int = 5,
    log_vars: list[str] | None = None,
) -> Any:
    """Build the rows=variables x cols=stratifiers grouped-boxplot figure.

    ``log_vars`` names variables whose row is drawn on a symmetric-log y-axis
    (defaults to any variable whose name contains ``"precip"``), so heavy-tailed
    precipitation scores do not collapse every box onto zero.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if log_vars is None:
        log_vars = [v for v in var_names if _is_precip(v)]
    log_set = set(log_vars)

    nrows = len(var_names)
    ncols = len(STRATIFIERS)
    color_of = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(model_order)}
    width = 0.8 / max(1, len(model_order))

    rc = {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(2.7 * ncols, 2.5 * nrows),
            squeeze=False,
            sharey="row",
        )
        model_arr = df["model"].to_numpy()
        for r, var in enumerate(var_names):
            metric_col = f"{metric}_{var}"
            metric_arr = df[metric_col].to_numpy(dtype=float)
            is_log = var in log_set
            linthresh = _symlog_linthresh(metric_arr) if is_log else None
            for c, strat in enumerate(STRATIFIERS):
                ax = axes[r][c]
                keys, present, label_for, _order_for = _bin_keys(df, strat, nbins)
                x = np.arange(len(present))
                if is_log:
                    ax.set_yscale("symlog", linthresh=linthresh, linscale=0.4)
                ax.grid(True, axis="y", which="major", lw=0.5, color="0.85")
                ax.set_axisbelow(True)
                for i, b in enumerate(present):
                    sel = keys == b
                    for k, model in enumerate(model_order):
                        m = sel & (model_arr == model)
                        vals = metric_arr[m]
                        vals = vals[np.isfinite(vals)]
                        if vals.size == 0:
                            continue
                        pos = x[i] + (k - (len(model_order) - 1) / 2) * width
                        bp = ax.boxplot(
                            vals,
                            positions=[pos],
                            widths=width * 0.9,
                            patch_artist=True,
                            showfliers=False,
                            manage_ticks=False,
                            medianprops=dict(color="white", linewidth=1.1),
                            whiskerprops=dict(color="0.3", linewidth=0.8),
                            capprops=dict(color="0.3", linewidth=0.8),
                            zorder=3,
                        )
                        for box in bp["boxes"]:
                            box.set_facecolor(color_of[model])
                            # White hairline separates adjacent grouped boxes.
                            box.set_edgecolor("white")
                            box.set_linewidth(0.8)
                ax.set_xticks(x)
                if r == nrows - 1:
                    ax.set_xticklabels(
                        [label_for[b] for b in present], rotation=30, ha="right"
                    )
                else:
                    ax.set_xticklabels([])
                if r == 0:
                    ax.set_title(_STRATIFIER_LABEL[strat])
                if c == 0:
                    ax.set_ylabel(f"{_display_var(var)} {metric.upper()}")
        # Clamp each log row just below zero (after all boxes are drawn, so this
        # wins over autoscale) to hide the near-zero noise band's downward tail.
        for r, var in enumerate(var_names):
            if var in log_set:
                col = df[f"{metric}_{var}"].to_numpy(dtype=float)
                axes[r][0].set_ylim(bottom=-_symlog_linthresh(col))

        proxies = [
            Patch(facecolor=color_of[m], label=_display_model(m)) for m in model_order
        ]
        # Reserve a top band for the legend so it never overlaps column titles.
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.legend(
            handles=proxies,
            loc="upper center",
            ncol=len(model_order),
            bbox_to_anchor=(0.5, 1.0),
            frameon=False,
        )
    return fig
