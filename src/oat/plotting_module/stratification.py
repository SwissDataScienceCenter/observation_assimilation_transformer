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
    if np.abs(coords[:, 0]).max() > 1e5:
        from pyproj import Transformer

        tf = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
        lon, lat = tf.transform(coords[:, 0], coords[:, 1])
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    return coords[:, 0], coords[:, 1]


def nearest_obs_distance(eval_coords: NDArray, obs_coords: NDArray) -> NDArray:
    """Min Euclidean distance from each eval station to any observed station.

    Distances are in the native coordinate units (metres for LV95, degrees for
    WGS84).
    """
    eval_coords = np.asarray(eval_coords, dtype=np.float64)
    obs_coords = np.asarray(obs_coords, dtype=np.float64)
    dists = np.linalg.norm(eval_coords[:, None, :] - obs_coords[None, :, :], axis=-1)
    return np.asarray(dists.min(axis=1))


def season_labels(times: Any) -> NDArray:
    """Map a sequence of timestamps to DJF/MAM/JJA/SON labels."""
    months = pd.DatetimeIndex(times).month
    return np.array([_MONTH_TO_SEASON[int(m)] for m in months], dtype=object)


def collect_station_covariates(
    dataset: Any, eval_indices: list[int], obs_indices: list[int]
) -> dict[str, NDArray]:
    """Compute static per-eval-station covariates for the long dump."""
    coords = dataset.station_coords.cpu().numpy()
    eval_coords = coords[eval_indices]
    obs_coords = coords[obs_indices]
    geo = dataset.station_geo.cpu().numpy()  # [N, 3] easting, northing, dem
    lon, lat = detect_lonlat(eval_coords)
    return {
        "station_idx": np.asarray(eval_indices),
        "station_x": eval_coords[:, 0],
        "station_y": eval_coords[:, 1],
        "lon": lon,
        "lat": lat,
        "elevation": geo[eval_indices, 2],
        "nearest_obs_dist": nearest_obs_distance(eval_coords, obs_coords),
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
    # Guard against all-NaN stratifier
    if np.isfinite(values).sum() == 0:
        return np.array([]), [], {}, {}
    idx, edges = _assign_fixed_width_bins(values, nbins)
    keys = idx.astype(int)
    present = sorted(set(int(i) for i in idx))
    label_for = {i: _bin_label(edges, i) for i in present}
    order_for = {i: i for i in present}
    return keys, present, label_for, order_for


def stratified_bar_stats(
    df: pd.DataFrame,
    stratifier: str,
    metric_col: str,
    model_col: str = "model",
    nbins: int = 5,
) -> pd.DataFrame:
    """Aggregate a metric into grouped-bar statistics per (bin, model).

    For ``stratifier == "season"`` the bins are the canonical DJF/MAM/JJA/SON
    categories (only those present). Otherwise the stratifier column is split
    into ``nbins`` fixed-width bins. Bar height is the NaN-ignoring mean of
    ``metric_col``; ``se`` is its standard error across the samples in the bin.
    """
    # Guard against empty input
    if len(df) == 0:
        return pd.DataFrame(columns=["bin", "bin_order", "model", "mean", "se", "n"])

    keys, present, label_for, order_for = _bin_keys(df, stratifier, nbins)
    if len(present) == 0:
        return pd.DataFrame(columns=["bin", "bin_order", "model", "mean", "se", "n"])

    models = list(df[model_col].unique())
    records: list[dict[str, Any]] = []
    for b in present:
        sel = keys == b
        for model in models:
            m = sel & (df[model_col].to_numpy() == model)
            vals = df.loc[m, metric_col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            se = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
            records.append(
                {
                    "bin": label_for[b],
                    "bin_order": order_for[b],
                    "model": model,
                    "mean": float(vals.mean()),
                    "se": se,
                    "n": int(vals.size),
                }
            )
    return (
        pd.DataFrame(records).sort_values(["bin_order", "model"]).reset_index(drop=True)
    )


STRATIFIERS = ["season", "elevation", "nearest_obs_dist", "lon", "lat"]

_STRATIFIER_LABEL = {
    "season": "season",
    "elevation": "elevation [m]",
    "nearest_obs_dist": "dist. to nearest obs",
    "lon": "longitude",
    "lat": "latitude",
}


def make_stratification_figure(
    df: pd.DataFrame,
    var_names: list[str],
    metric: str,
    model_order: list[str],
    nbins: int = 5,
) -> Any:
    """Build the rows=variables x cols=stratifiers grouped-boxplot figure."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    nrows = len(var_names)
    ncols = len(STRATIFIERS)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_of = {m: colors[i % len(colors)] for i, m in enumerate(model_order)}
    width = 0.8 / max(1, len(model_order))

    for r, var in enumerate(var_names):
        metric_col = f"{metric}_{var}"
        for c, strat in enumerate(STRATIFIERS):
            ax = axes[r][c]
            keys, present, label_for, _order_for = _bin_keys(df, strat, nbins)
            x = np.arange(len(present))
            model_arr = df["model"].to_numpy()
            metric_arr = df[metric_col].to_numpy(dtype=float)
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
                    )
                    for box in bp["boxes"]:
                        box.set_facecolor(color_of[model])
            ax.set_xticks(x)
            ax.set_xticklabels(
                [label_for[b] for b in present], rotation=45, ha="right", fontsize=8
            )
            if r == 0:
                ax.set_title(_STRATIFIER_LABEL[strat])
            if c == 0:
                ax.set_ylabel(f"{metric.upper()} — {var}")
    proxies = [Patch(facecolor=color_of[m], label=m) for m in model_order]
    fig.legend(handles=proxies, loc="upper right", ncol=len(model_order))
    fig.tight_layout()
    return fig
