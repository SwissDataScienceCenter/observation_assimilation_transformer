"""Two- or three-panel overview of station splits, CosmoPeakWeather & ERA5-Meteostat.

Left panel:   CosmoPeakWeather — Switzerland, Swiss LV95, plain axes (metric → equal aspect).
Middle panel: ERA5-Meteostat  — Europe, Lambert Conformal projection via cartopy.
Right panel (optional, --region-holdout): ERA5-Meteostat — same projection, colored
    by the region-holdout split (train / val / test-context / test-eval) instead.

The first two panels mark train (white squares), val (grey triangles), test
(black circles) stations and draw a dashed rectangle around the NWP data domain.
The optional third panel marks train, val, test-eval the same way, adds an
orange diamond for test-context stations (revealed as context at test time),
and draws the region bounding box instead of the NWP domain box.

Usage:
    python -m scripts.plot_dataset_stations
    python -m scripts.plot_dataset_stations --out data/figures/dataset_stations.pdf
    python -m scripts.plot_dataset_stations --seed 0 --dpi 200 --out my_plot.png
    python -m scripts.plot_dataset_stations --region-holdout
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from oat.data_module.data_split import create_splits
from oat.data_module.setup import _instantiate_dataset
from oat.plotting_module.geo_overlays import _add_borders

_CFG_DIR = Path(__file__).resolve().parent.parent / "configs"
_NE_STYLE = dict(edgecolor="black", linewidth=0.4, facecolor="none", zorder=3)
_MS = 1.5      # marker size
_MLW = 0.1  # marker edge linewidth
# Switzerland panel box (LV95 m): min (x, y), max (x, y). This box is the
# extent of the paper figure. It is the area of the old COSMO reference grid.
_COSMO_BOX_LV95 = ((2255500.0, 840500.0), (2964500.0, 1479500.0))


def _load_cfg(family: str) -> DictConfig:
    """Compose the root config with the ``<family>/temp_precip`` dataset."""
    with initialize_config_dir(config_dir=str(_CFG_DIR), version_base="1.3"):
        return compose(
            config_name="config", overrides=[f"dataset={family}/temp_precip"]
        )


def _markers(ax, coords, train_idx, val_idx, test_idx, *, transform=None):
    kw: dict = {} if transform is None else {"transform": transform}
    ax.plot(
        coords[train_idx, 0], coords[train_idx, 1],
        "s", color="C0", markeredgecolor="C0", markeredgewidth=_MLW,
        markersize=_MS, zorder=5, label=f"Train ({len(train_idx)})", **kw,
    )
    ax.plot(
        coords[val_idx, 0], coords[val_idx, 1],
        "^", color="C2", markeredgecolor="C2", markeredgewidth=_MLW,
        markersize=_MS, zorder=6, label=f"Val ({len(val_idx)})", **kw,
    )
    ax.plot(
        coords[test_idx, 0], coords[test_idx, 1],
        "o", color="C3", markeredgecolor="C3", markeredgewidth=_MLW,
        markersize=_MS, zorder=7, label=f"Test ({len(test_idx)})", **kw,
    )


def _markers_region_holdout(
    ax, coords, train_idx, val_idx, test_context_idx, test_eval_idx, *, transform=None,
):
    """Plot the region-holdout split: train, val, test-context, test-eval."""
    kw: dict = {} if transform is None else {"transform": transform}
    ax.plot(
        coords[train_idx, 0], coords[train_idx, 1],
        "s", color="C0", markeredgecolor="C0", markeredgewidth=_MLW,
        markersize=_MS, zorder=5, label=f"Train ({len(train_idx)})", **kw,
    )
    ax.plot(
        coords[val_idx, 0], coords[val_idx, 1],
        "^", color="C2", markeredgecolor="C2", markeredgewidth=_MLW,
        markersize=_MS, zorder=6, label=f"Val ({len(val_idx)})", **kw,
    )
    ax.plot(
        coords[test_context_idx, 0], coords[test_context_idx, 1],
        "D", color="C1", markeredgecolor="C1", markeredgewidth=_MLW,
        markersize=_MS, zorder=7, label=f"Test-context ({len(test_context_idx)})", **kw,
    )
    ax.plot(
        coords[test_eval_idx, 0], coords[test_eval_idx, 1],
        "o", color="C3", markeredgecolor="C3", markeredgewidth=_MLW,
        markersize=_MS, zorder=8, label=f"Test-eval ({len(test_eval_idx)})", **kw,
    )


def _load_region_holdout_indices(
    split_path: Path, stations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve the region-holdout split's station IDs to positions.

    Parameters
    ----------
    split_path : Path
        Path to the region-holdout split YAML (station-ID lists).
    stations : np.ndarray
        ERA5-Meteostat dataset station IDs, in ``station_coords`` order.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Position arrays: train_idx, val_idx, test_context_idx, test_eval_idx.
    """
    split_cfg = OmegaConf.load(split_path)
    id_to_pos = {sid: i for i, sid in enumerate(stations)}

    def _resolve(ids: list[str]) -> np.ndarray:
        return np.array([id_to_pos[sid] for sid in ids], dtype=int)

    return (
        _resolve(split_cfg.train_station_indices),
        _resolve(split_cfg.val_station_indices),
        _resolve(split_cfg.test_context_station_indices),
        _resolve(split_cfg.test_station_indices),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/figures/dataset_stations.png",
                        help="Output path; extension sets format (.png / .pdf / .svg).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible train/val/test split.")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--region-holdout", action="store_true",
                        help="Add a third panel showing the region-holdout split.")
    parser.add_argument("--region-holdout-split",
                        default="configs/split/spatial_region_holdout.yaml",
                        help="Path to the region-holdout split YAML.")
    parser.add_argument("--region-lat-min", type=float, default=43.8)
    parser.add_argument("--region-lat-max", type=float, default=49.2)
    parser.add_argument("--region-lon-min", type=float, default=3.3)
    parser.add_argument("--region-lon-max", type=float, default=11.8)
    args = parser.parse_args()

    # ── Load datasets & compute splits ───────────────────────────────────────
    print("Loading CosmoPeakWeather dataset …")
    cosmo_ds = _instantiate_dataset(_load_cfg("cosmo_peakweather"))
    cosmo_train, cosmo_val, cosmo_test = create_splits(cosmo_ds, seed=args.seed)["station_indices"]
    cosmo_coords = cosmo_ds.station_coords.numpy()  # (N, 2): easting, northing  [m LV95]

    print("Loading ERA5-Meteostat dataset …")
    era5_cfg = _load_cfg("era5_meteostat")
    era5_ds = _instantiate_dataset(era5_cfg)
    era5_train, era5_val, era5_test = create_splits(era5_ds, seed=args.seed)["station_indices"]
    era5_coords = era5_ds.station_coords.numpy()    # (N, 2): longitude, latitude  [° WGS84]

    if args.region_holdout:
        print("Loading region-holdout split …")
        rh_train, rh_val, rh_test_context, rh_test_eval = _load_region_holdout_indices(
            Path(args.region_holdout_split), era5_ds.stations,
        )

    # ── COSMO domain bounds (LV95 m) ──────────────────────────────────────────
    (cx0, cy0), (cx1, cy1) = _COSMO_BOX_LV95
    c_pad_x = (cx1 - cx0) * 0.04
    c_pad_y = (cy1 - cy0) * 0.04

    # ── ERA5 domain bounds (WGS84) ────────────────────────────────────────────
    ex0 = float(era5_cfg.dataset.nwp_lon_min); ex1 = float(era5_cfg.dataset.nwp_lon_max)
    ey0 = float(era5_cfg.dataset.nwp_lat_min); ey1 = float(era5_cfg.dataset.nwp_lat_max)

    # ── Figure: plain axes (left) + cartopy axes (middle[, right]) ───────────
    proj = ccrs.LambertConformal(central_longitude=10.0, central_latitude=52.0)
    if args.region_holdout:
        fig = plt.figure(figsize=(21, 5.5))
        gs = gridspec.GridSpec(
            1, 3, figure=fig, width_ratios=[1, 2.5, 2.5], wspace=0.08,
        )
    else:
        fig = plt.figure(figsize=(15, 5.5))
        gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 2.5], wspace=0.08)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], projection=proj)
    ax2 = fig.add_subplot(gs[2], projection=proj) if args.region_holdout else None

    # ── Left: CosmoPeakWeather (LV95 plain axes) ──────────────────────────────
    cosmo_xs = np.linspace(cx0, cx1, 100)
    cosmo_ys = np.linspace(cy0, cy1, 100)
    _add_borders([ax0], cosmo_xs, cosmo_ys)

    ax0.plot(
        [cx0, cx1, cx1, cx0, cx0], [cy0, cy0, cy1, cy1, cy0],
        "--", color="black", linewidth=0.8, zorder=4, dash_capstyle="butt",
    )
    _markers(ax0, cosmo_coords, cosmo_train, cosmo_val, cosmo_test)

    ax0.set_xlim(cx0 - c_pad_x, cx1 + c_pad_x)
    ax0.set_ylim(cy0 - c_pad_y, cy1 + c_pad_y)
    ax0.set_aspect("equal")
    ax0.set_xlabel("Easting [m]", fontsize=9)
    ax0.set_ylabel("Northing [m]", fontsize=9)
    ax0.set_title("Switzerland dataset", fontsize=10)
    ax0.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax0.tick_params(labelsize=7)
    ax0.ticklabel_format(style="sci", axis="both", scilimits=(6, 6))

    # ── Right: ERA5-Meteostat (Lambert Conformal via cartopy) ─────────────────
    ax1.add_feature(cfeature.NaturalEarthFeature("physical", "coastline", "10m", **_NE_STYLE))
    ax1.add_feature(cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land", "10m", **_NE_STYLE))
    ax1.add_feature(cfeature.NaturalEarthFeature("physical", "lakes", "10m", **_NE_STYLE))
    ax1.set_facecolor("white")

    ax1.set_extent([ex0 - 3, ex1 + 3, ey0 - 1, ey1 + 1], crs=ccrs.PlateCarree())

    # ERA5 domain rectangle — densify edges so top/bottom arc correctly in LCC
    _n2 = 200
    _era5_box = np.vstack([
        np.column_stack([np.linspace(ex0, ex1, _n2), np.full(_n2, ey1)]),
        np.column_stack([np.full(_n2, ex1),           np.linspace(ey1, ey0, _n2)]),
        np.column_stack([np.linspace(ex1, ex0, _n2), np.full(_n2, ey0)]),
        np.column_stack([np.full(_n2, ex0),           np.linspace(ey0, ey1, _n2)]),
        [[ex0, ey1]],
    ])
    ax1.plot(_era5_box[:, 0], _era5_box[:, 1], "--", color="black", linewidth=0.8,
             transform=ccrs.PlateCarree(), zorder=4, dash_capstyle="butt")

    # Region-holdout box (same box as the optional third panel)
    rx0, rx1 = args.region_lon_min, args.region_lon_max
    ry0, ry1 = args.region_lat_min, args.region_lat_max
    ax1.plot(
        [rx0, rx1, rx1, rx0, rx0], [ry0, ry0, ry1, ry1, ry0],
        "--", color="black", linewidth=1.0, transform=ccrs.PlateCarree(),
        zorder=10, dash_capstyle="butt",
    )

    # One label per edge of the ERA5 domain box, placed at the midpoint,
    # showing the coordinate value that defines that edge.
    _geo = ccrs.PlateCarree()
    _tkw = dict(transform=_geo, fontsize=7, clip_on=False)
    _mid_lon = (ex0 + ex1) / 2
    _mid_lat = (ey0 + ey1) / 2
    _fmt_lon = lambda v: f"{abs(v):g}°W" if v < 0 else f"{v:g}°E"
    ax1.text(ex0, _mid_lat, _fmt_lon(ex0), ha="right",  va="center", **_tkw)
    ax1.text(ex1, _mid_lat, _fmt_lon(ex1), ha="left",   va="center", **_tkw)
    ax1.text(_mid_lon, ey0,  f"{ey0:g}°N", ha="center", va="top",    **_tkw)
    ax1.text(_mid_lon, ey1,  f"{ey1:g}°N", ha="center", va="bottom",  **_tkw)

    _markers(ax1, era5_coords, era5_train, era5_val, era5_test,
             transform=ccrs.PlateCarree())

    ax1.set_title("Europe dataset", fontsize=10)
    ax1.legend(loc="lower left", fontsize=8, framealpha=0.9)

    # ── Optional third panel: ERA5-Meteostat region-holdout split ────────────
    if ax2 is not None:
        ax2.add_feature(cfeature.NaturalEarthFeature("physical", "coastline", "10m", **_NE_STYLE))
        ax2.add_feature(cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land", "10m", **_NE_STYLE))
        ax2.add_feature(cfeature.NaturalEarthFeature("physical", "lakes", "10m", **_NE_STYLE))
        ax2.set_facecolor("white")
        ax2.set_extent([ex0 - 3, ex1 + 3, ey0 - 1, ey1 + 1], crs=ccrs.PlateCarree())

        ax2.plot(_era5_box[:, 0], _era5_box[:, 1], "--", color="black", linewidth=0.8,
                 transform=ccrs.PlateCarree(), zorder=4, dash_capstyle="butt")

        ax2.text(ex0, _mid_lat, _fmt_lon(ex0), ha="right",  va="center", **_tkw)
        ax2.text(ex1, _mid_lat, _fmt_lon(ex1), ha="left",   va="center", **_tkw)
        ax2.text(_mid_lon, ey0,  f"{ey0:g}°N", ha="center", va="top",    **_tkw)
        ax2.text(_mid_lon, ey1,  f"{ey1:g}°N", ha="center", va="bottom",  **_tkw)

        ax2.plot(
            [rx0, rx1, rx1, rx0, rx0], [ry0, ry0, ry1, ry1, ry0],
            "--", color="black", linewidth=1.0, transform=ccrs.PlateCarree(),
            zorder=10, dash_capstyle="butt",
        )

        _markers_region_holdout(
            ax2, era5_coords, rh_train, rh_val, rh_test_context, rh_test_eval,
            transform=ccrs.PlateCarree(),
        )

        ax2.set_title("Europe dataset — region holdout", fontsize=10)
        ax2.legend(loc="lower left", fontsize=8, framealpha=0.9)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=args.dpi, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
