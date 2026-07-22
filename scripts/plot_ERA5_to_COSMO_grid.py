"""Plot OAT predictions over the full COSMO-1E grid.

Loads the 919 620 native COSMO-1E grid points from the anemoi dataset and
runs batched inference with a checkpoint (typically ERA5/MeteoStat-trained),
bypassing the center+extent approach of plot_oat_example.py.

Usage
-----
    python -m scripts.plot_oat_nwp_grid <checkpoint> --var temperature --time 2024-06-21:12
    python -m scripts.plot_oat_nwp_grid <checkpoint> --batch-size 20000
"""
# Variables stored in Kelvin that need conversion to Celsius for plotting
_KELVIN_VARS = {"T_G", "T_2M", "2t"}
# ERA5 variables stored in metres that need conversion to mm for plotting
_M_TO_MM_VARS = {"tp", "TOT_PREC_1H"}

import argparse
import logging
import sys
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from anemoi.datasets import open_dataset
from omegaconf import OmegaConf

from oat.data_module.cosmo_peakweather_dataset import CosmoPeakWeatherDataset
from oat.data_module.era5_meteostat_dataset import DERIVED_FEATURES, ERA5MeteostatDataset
from oat.data_module.setup import setup_data_and_transform
from oat.plotting_module import OATPlotter
from oat.plotting_module.geo_overlays import _add_borders
from oat.predictor_module.checkpoint_utils import load_checkpoint
from oat.predictor_module.oat_predictor import OATPredictor

OUT_DIR = Path(__file__).parents[1] / "data/figures"
_PROJECT_ROOT = Path(__file__).parents[1]
_PLOT_CONFIG_PATHS: dict[type, Path] = {
    ERA5MeteostatDataset: _PROJECT_ROOT / "configs/dataset/era5_meteostat/plot_config.yaml",
    CosmoPeakWeatherDataset: _PROJECT_ROOT / "configs/dataset/cosmo_peakweather/plot_config.yaml",
}
DEFAULT_ANEMOI = (
    "/store_new/mch/msopr/ml/datasets/"
    "mch-co1e-an-archive-0p01-2019-2024-1h-v1-pl13.zarr"
)
DEFAULT_NWP_EXTRACTED = (
    "/scratch/mch/cdonner/data/obs_downscaling/raw/cosmo-1e_rechunked/nwp_extracted.zarr"
)


def _load_plot_config(dataset):
    path = _PLOT_CONFIG_PATHS.get(type(dataset))
    if path and path.exists():
        return OmegaConf.load(path).obs_plot_config
    return None


# TODO should move these in some common utils for the scripts
def _resolve_time_idx(dataset, time_str: str) -> int:
    ts = pd.Timestamp(time_str.replace(":", " "), tz="UTC")
    idx = int(dataset.times.searchsorted(ts))
    if idx >= len(dataset) or dataset.times[idx] != ts:
        raise ValueError(
            f"Timestamp {ts} not found in dataset. "
            f"Range: {dataset.times[0]} – {dataset.times[-1]}"
        )
    return idx


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot OAT predictions at every COSMO-1E grid point."
    )
    p.add_argument("checkpoint", help="Path to .ckpt file")
    p.add_argument(
        "--anemoi",
        default=DEFAULT_ANEMOI,
        metavar="PATH",
        help=f"Anemoi dataset to read grid coordinates from (default: {DEFAULT_ANEMOI})",
    )
    p.add_argument("--var", default=None, help="Variable to plot (default: first in predictor).")
    p.add_argument(
        "--time",
        default=None,
        metavar="YYYY-MM-DD[:HH]",
        help="Snapshot time. Default: first validation timestep.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        metavar="N",
        help="Query points per inference call (default: 50000).",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=100,
        metavar="N",
        help="Samples from the predictive distribution (default: 100).",
    )
    p.add_argument(
        "--nwp-zarr",
        default=DEFAULT_NWP_EXTRACTED,
        metavar="PATH",
        help=f"nwp_extracted.zarr for COSMO ground truth (default: {DEFAULT_NWP_EXTRACTED})",
    )
    p.add_argument(
        "--nwp-var",
        default=None,
        metavar="NAME",
        help=(
            "COSMO variable name to show as ground truth (e.g. T_2M, U_10M). "
            "Available: T_G, T_2M, U_10M, V_10M, TOT_PREC_1H, CLCT. "
            "If omitted, no ground-truth panel is added."
        ),
    )
    p.add_argument(
        "--era5-var",
        default=None,
        metavar="NAME",
        help=(
            "ERA5 source variable to show at native resolution (e.g. 2t, 10u, 10v), "
            "clipped to the COSMO domain. If omitted, no ERA5 panel is added."
        ),
    )
    p.add_argument(
        "--split",
        choices=["val", "test"],
        default=None,
        help=(
            "Show station split mode. 'val': observed (train) stations as circles, "
            "validation stations as squares. 'test': observed (train+val) as circles, "
            "test stations as squares."
        ),
    )
    p.add_argument(
        "--no-stations",
        action="store_true",
        default=False,
        help="Suppress station markers.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger(__name__)

    args = _parse_args()

    predictor, cfg = load_checkpoint(OATPredictor, args.checkpoint, map_location="cpu")
    cfg.dataset.data_points_for_stats = 0
    predictor = predictor.eval()

    # TODO: move shared script utilities (split resolution, time idx, plot config) to scripts/utils.py
    dataset, _, _, val_loader, test_loader, *_ = setup_data_and_transform(cfg, logger)

    obs_station_indices = None
    eval_indices = None
    if args.split is not None:
        split_loader = val_loader if args.split == "val" else test_loader
        split_dataset = split_loader.dataset
        obs_station_indices = list(split_dataset.observed_indices)  # type: ignore[attr-defined]
        eval_indices = list(split_dataset.eval_indices)  # type: ignore[attr-defined]

    # Load COSMO grid coordinates from anemoi
    logger.info("Loading COSMO grid from %s", args.anemoi)
    anemoi_ds = open_dataset(args.anemoi)
    grid_shape = anemoi_ds.field_shape          # (786, 1170)
    lats = anemoi_ds.latitudes                  # (919620,)
    lons = anemoi_ds.longitudes                 # (919620,)
    anemoi_dates = anemoi_ds.dates              # (45378,) numpy.datetime64
    # ERA5MeteostatDataset uses (lon, lat) order
    query_coords = torch.tensor(
        np.stack([lons, lats], axis=1), dtype=torch.float32
    )
    logger.info(
        "Grid: %d × %d = %d query points  |  lon [%.2f, %.2f]  lat [%.2f, %.2f]",
        grid_shape[0], grid_shape[1], len(query_coords),
        lons.min(), lons.max(), lats.min(), lats.max(),
    )

    var = args.var if args.var is not None else next(iter(predictor.var_to_indx))
    if var not in predictor.var_to_indx:
        logger.error("Variable '%s' not in checkpoint. Available: %s", var, list(predictor.var_to_indx))
        sys.exit(1)
    var_idx = predictor.var_to_indx[var]

    if args.time is not None:
        snap_idx = _resolve_time_idx(dataset, args.time)
    else:
        snap_idx = (
            val_loader.dataset.indices[0]
            if hasattr(val_loader.dataset, "indices")
            else 0
        )
    snap_time_iso = dataset.times[snap_idx].strftime("%Y-%m-%dT%H")
    logger.info("Snapshot: %s  |  var: %s", snap_time_iso, var)

    plotter = OATPlotter(
        predictor=predictor,
        dataset=dataset,
        obs_station_indices=obs_station_indices,
        eval_indices=eval_indices,
        n_samples=args.n_samples,
        device="cpu",
        plot_config=_load_plot_config(dataset),
    )

    # Batched inference
    N = len(query_coords)
    means = []
    for start in range(0, N, args.batch_size):
        end = min(start + args.batch_size, N)
        samples = plotter._run_inference(snap_idx, query_coords[start:end])  # [B, n_samples, n_vars]
        var_samples = samples[:, :, var_idx]
        means.append(var_samples.mean(dim=1).detach().cpu().numpy())
        logger.info("Inference: %d / %d", end, N)

    mean_field = np.concatenate(means).reshape(grid_shape)
    lon_grid = lons.reshape(grid_shape)
    lat_grid = lats.reshape(grid_shape)

    # Optionally load COSMO ground truth from nwp_extracted.zarr
    cosmo_field = None
    if args.nwp_var is not None:
        import zarr as _zarr
        nwp_store = _zarr.open(args.nwp_zarr, "r")
        cosmo_vars = list(nwp_store.attrs["variables"])
        _COSMO_WIND_SPEED = "wind_speed"
        _cosmo_missing = (
            args.nwp_var not in cosmo_vars
            and not (args.nwp_var == _COSMO_WIND_SPEED and {"U_10M", "V_10M"}.issubset(cosmo_vars))
        )
        if _cosmo_missing:
            logger.warning("--nwp-var '%s' not in %s (available: %s); skipping GT panel.",
                           args.nwp_var, args.nwp_zarr, cosmo_vars)
        else:
            snap_ts = np.datetime64(dataset.times[snap_idx].isoformat())
            anemoi_idx = int(np.searchsorted(anemoi_dates, snap_ts))
            if args.nwp_var == _COSMO_WIND_SPEED:
                u = nwp_store[anemoi_idx, cosmo_vars.index("U_10M"), :].reshape(grid_shape)
                v = nwp_store[anemoi_idx, cosmo_vars.index("V_10M"), :].reshape(grid_shape)
                cosmo_field = np.sqrt(u ** 2 + v ** 2)
            else:
                cosmo_field = nwp_store[anemoi_idx, cosmo_vars.index(args.nwp_var), :].reshape(grid_shape)
                if args.nwp_var in _KELVIN_VARS:
                    cosmo_field -= 273.15
                if args.nwp_var in _M_TO_MM_VARS:
                    cosmo_field *= 1000.0
            logger.info(
                "Loaded COSMO %s at anemoi index %d  |  range [%.4g, %.4g]",
                args.nwp_var, anemoi_idx, float(cosmo_field.min()), float(cosmo_field.max()),
            )

    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())

    # Optionally load ERA5 on the reference regular grid (same resampling as __getitem__)
    era5_field_2d = era5_lon_grid = era5_lat_grid = None
    if args.era5_var is not None:
        is_source = args.era5_var in dataset._source_vars
        is_derived = args.era5_var in DERIVED_FEATURES
        if not is_source and not is_derived:
            logger.warning(
                "--era5-var '%s' not in source vars or derived features (available: %s + %s); skipping.",
                args.era5_var, list(dataset._source_vars), list(DERIVED_FEATURES),
            )
        else:
            nwp_idx = int(dataset.time_indices[snap_idx])
            raw = np.array(dataset.nwp_ds[nwp_idx][dataset.nwp_var_idx])
            raw = raw.reshape(len(dataset.nwp_var_idx), -1)
            raw_grid = raw[:, dataset.nwp_indices].reshape(
                len(dataset.nwp_var_idx), *dataset.nwp_shape
            )  # (n_vars, n_ref_lats, n_ref_lons)
            src = {v: torch.from_numpy(raw_grid[i]) for i, v in enumerate(dataset._source_vars)}
            if is_derived:
                deps, fn = DERIVED_FEATURES[args.era5_var]
                era5_field_2d = fn(*[src[d] for d in deps]).numpy()
            else:
                era5_field_2d = raw_grid[list(dataset._source_vars).index(args.era5_var)]
            if args.era5_var in _KELVIN_VARS:
                era5_field_2d -= 273.15
            if args.era5_var in _M_TO_MM_VARS:
                era5_field_2d *= 1000.0
            era5_lon_grid, era5_lat_grid = np.meshgrid(dataset._ref_lons, dataset._ref_lats)
            logger.info(
                "ERA5 %s on reference grid: %d × %d",
                args.era5_var, *dataset.nwp_shape,
            )

    import matplotlib.colors as mcolors
    var_cfg = plotter._get_var_plot_config(var)
    cmap = var_cfg.get("cmap", "viridis")
    ckpt_tag = Path(args.checkpoint).parent.name

    # Shared colour range across OAT mean, COSMO, ERA5
    all_fields = [mean_field]
    if cosmo_field is not None:
        all_fields.append(cosmo_field)
    if era5_field_2d is not None:
        all_fields.append(era5_field_2d)
    all_vals = np.concatenate([f.ravel() for f in all_fields])
    cfg_vmin = var_cfg.get("vmin")
    cfg_vmax = var_cfg.get("vmax")
    shared_vmin = float(cfg_vmin if cfg_vmin is not None else np.nanpercentile(all_vals, 1.0))
    shared_vmax = float(cfg_vmax if cfg_vmax is not None else np.nanpercentile(all_vals, 99))
    if shared_vmax - shared_vmin < 1e-6:
        shared_vmax = shared_vmin + 1.0
    logger.info("Shared colour range: %.4g – %.4g", shared_vmin, shared_vmax)
    shared_norm = mcolors.Normalize(vmin=shared_vmin, vmax=shared_vmax)
    cb_label = "°C" if (args.nwp_var in _KELVIN_VARS or args.era5_var in _KELVIN_VARS) else var

    # Single-row layout: ERA5 | OAT | COSMO, sharing one colorbar.
    proj = ccrs.PlateCarree()
    panels = []
    if era5_field_2d is not None:
        panels.append(
            (era5_lon_grid, era5_lat_grid, era5_field_2d, f"ERA5 {args.era5_var}", False)
        )
    panels.append((lon_grid, lat_grid, mean_field, "OAT — predictive mean", True))
    if cosmo_field is not None:
        panels.append((lon_grid, lat_grid, cosmo_field, f"COSMO {args.nwp_var}", False))

    fig, axs = plt.subplots(
        1, len(panels), figsize=(6 * len(panels), 5), layout="constrained",
        subplot_kw={"projection": proj}, squeeze=False,
    )
    axs = list(axs[0])
    # fig.suptitle(f"{ckpt_tag} — {var}  ({snap_time_iso})")

    mesh = None
    for ax, (xg, yg, field, title, is_oat) in zip(axs, panels):
        m = ax.pcolormesh(
            xg, yg, field, cmap=cmap, norm=shared_norm, shading="auto",
            transform=proj, rasterized=True,  # keep PDF small; vector axes/text stay crisp
        )
        ax.set_title(title)
        if is_oat:
            mesh = m
            if not args.no_stations:
                plotter._overlay_stations(ax, snap_idx, var_idx, mesh=m)
                ax.legend(fontsize=7)

    # cos(lat) aspect: 1° lon is shorter than 1° lat at mid-latitudes, so equal-
    # degree PlateCarree over-widens the map. This restores faithful proportions.
    mean_lat = 0.5 * (lat_min + lat_max)
    for i, ax in enumerate(axs):
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
        ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = i == 0

    # One shared colorbar for all panels (identical norm + cmap).
    fig.colorbar(mesh, ax=axs, location="right", shrink=0.8, label=cb_label)

    try:
        _add_borders(axs, lon_grid[0], lat_grid[:, 0])
    except Exception as exc:
        logger.warning("Geo overlays skipped: %s", exc)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{ckpt_tag}_{var}_{snap_time_iso}_nwp_grid"
    for ext in ("png", "pdf"):
        out_path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        logger.info("Saved → %s", out_path)
    plt.close(fig)


if __name__ == "__main__":
    print("Running OAT NWP-grid diagnostic...")
    main()
