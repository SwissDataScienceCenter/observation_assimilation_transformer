"""Two-panel overview of station splits for CosmoPeakWeather and ERA5-Meteostat.

Left panel:  CosmoPeakWeather — Switzerland, Swiss LV95, plain axes (metric → equal aspect).
Right panel: ERA5-Meteostat  — Europe, Lambert Conformal projection via cartopy.

Both panels mark train (white squares), val (grey triangles), test (black circles)
stations and draw a dashed rectangle around the NWP data domain.

Usage:
    python -m scripts.plot_dataset_stations
    python -m scripts.plot_dataset_stations --out data/figures/dataset_stations.pdf
    python -m scripts.plot_dataset_stations --seed 0 --dpi 200 --out my_plot.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from pyproj import Transformer
from shapely.geometry import Polygon

from oat.data_module.legacy.cosmo_peakweather_dataset import CosmoPeakWeatherDataset
from oat.data_module.legacy.data_split import create_splits
from oat.data_module.legacy.era5_meteostat_dataset import ERA5MeteostatDataset
from oat.plotting_module.geo_overlays import _add_borders

_CFG_DIR = Path(__file__).parent.parent / "configs" / "dataset"
_NE_STYLE = dict(edgecolor="black", linewidth=0.4, facecolor="none", zorder=3)
_MS = 1.5      # marker size
_MLW = 0.1  # marker edge linewidth


def _load_cosmo_cfg() -> OmegaConf:
    cfg_dir = _CFG_DIR / "cosmo_peakweather"
    cfg = OmegaConf.merge(
        OmegaConf.load(cfg_dir / "settings.yaml"),
        OmegaConf.create({k: v for k, v in OmegaConf.load(cfg_dir / "temp_precip.yaml").items()
                          if k != "defaults"}),
    )
    cfg.catalog = OmegaConf.load(cfg_dir / "catalog.yaml")
    return cfg


def _load_era5_cfg() -> OmegaConf:
    cfg_dir = _CFG_DIR / "era5_meteostat"
    cfg = OmegaConf.merge(
        OmegaConf.load(cfg_dir / "settings.yaml"),
        OmegaConf.create({k: v for k, v in OmegaConf.load(cfg_dir / "temp_precip.yaml").items()
                          if k != "defaults"}),
    )
    cfg.catalog = OmegaConf.load(cfg_dir / "catalog.yaml")
    return cfg


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/figures/dataset_stations.png",
                        help="Output path; extension sets format (.png / .pdf / .svg).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible train/val/test split.")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    # ── Load datasets & compute splits ───────────────────────────────────────
    print("Loading CosmoPeakWeather dataset …")
    cosmo_ds = CosmoPeakWeatherDataset(_load_cosmo_cfg())
    cosmo_train, cosmo_val, cosmo_test = create_splits(cosmo_ds, seed=args.seed)["station_indices"]
    cosmo_coords = cosmo_ds.station_coords.numpy()  # (N, 2): easting, northing  [m LV95]

    print("Loading ERA5-Meteostat dataset …")
    era5_ds = ERA5MeteostatDataset(_load_era5_cfg())
    era5_train, era5_val, era5_test = create_splits(era5_ds, seed=args.seed)["station_indices"]
    era5_coords = era5_ds.station_coords.numpy()    # (N, 2): longitude, latitude  [° WGS84]

    # ── COSMO domain bounds (LV95 m) ──────────────────────────────────────────
    cx0 = float(cosmo_ds.nwp_extent["min"][0]); cx1 = float(cosmo_ds.nwp_extent["max"][0])
    cy0 = float(cosmo_ds.nwp_extent["min"][1]); cy1 = float(cosmo_ds.nwp_extent["max"][1])
    c_pad_x = (cx1 - cx0) * 0.04
    c_pad_y = (cy1 - cy0) * 0.04

    # ── ERA5 domain bounds (WGS84) ────────────────────────────────────────────
    ex0 = float(era5_ds._ref_lons.min()); ex1 = float(era5_ds._ref_lons.max())
    ey0 = float(era5_ds._ref_lats.min()); ey1 = float(era5_ds._ref_lats.max())

    # ── COSMO domain in WGS84 (for inset box on Europe panel) ────────────────
    # Densify the LV95 boundary before reprojecting so the polygon is accurate.
    _tf = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    _n = 60
    _be = np.concatenate([np.linspace(cx0, cx1, _n), np.full(_n, cx1),
                          np.linspace(cx1, cx0, _n), np.full(_n, cx0)])
    _bn = np.concatenate([np.full(_n, cy1), np.linspace(cy1, cy0, _n),
                          np.full(_n, cy0), np.linspace(cy0, cy1, _n)])
    _blon, _blat = _tf.transform(_be, _bn)
    cosmo_wgs84_poly = Polygon(zip(_blon, _blat))

    # ── Figure: plain axes (left) + cartopy axes (right) ─────────────────────
    proj = ccrs.LambertConformal(central_longitude=10.0, central_latitude=52.0)
    fig = plt.figure(figsize=(15, 5.5))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 2.5], wspace=0.08)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], projection=proj)

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

    # COSMO-1e domain inset (black dashed) — already densified in LV95→WGS84 step
    _cosmo_box = np.array(cosmo_wgs84_poly.exterior.coords)
    ax1.plot(_cosmo_box[:, 0], _cosmo_box[:, 1], "--", color="black", linewidth=1.0,
             transform=ccrs.PlateCarree(), zorder=10, dash_capstyle="butt")

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

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=args.dpi, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
