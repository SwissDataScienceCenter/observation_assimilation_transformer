"""Plot OAT predictions over the full COSMO-1E grid.

Loads the 919 620 native COSMO-1E grid points from the anemoi dataset and
runs batched inference with a checkpoint (typically ERA5/MeteoStat-trained),
bypassing the center+extent approach of plot_oat_example.py.

Usage
-----
    python -m scripts.plot_ERA5_to_COSMO_grid 
        /scratch/mch/cdonner/observation_assimilation_transformer/checkpoints/oat/oat_era5_meteostat-temp_precip_20260720_job4889366/best-model-epoch=84-val_loss=0.2154.ckpt \
        --uninformed-checkpoint /scratch/mch/cdonner/observation_assimilation_transformer/checkpoints/uninformed_oat/uninformed_oat_era5_meteostat-temp_precip_20260715_job4849165/best-model-epoch=51-val_loss=0.2598.ckpt \
        --var temp --time 2024-06-21:12 --nwp-var T_2M --era5-var 2t --split test
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

# ---- Figure sizing & fonts (single source of truth; mirror in other paper figures) ----
# A fixed width makes the output size predictable and lets every figure share one font
# size: font points are absolute, so identical width + identical rcParams + the same
# include-width in LaTeX give identical on-page fonts. The figure HEIGHT is derived from
# the map aspect so the aspect-locked maps still fill their cells; the margins are fixed
# inches because they track the font size, not the figure size.
FIG_WIDTH_IN = 16.0   # target figure width (inches) — keep identical across figures
_MARGIN_W_IN = 2.3    # inches for left tick labels + the right-hand colorbar column
_MARGIN_H_IN = 1.4    # inches for panel titles + top/bottom margins
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
})

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
    p.add_argument("checkpoint", help="Path to .ckpt file (the OAT model)")
    p.add_argument(
        "--uninformed-checkpoint",
        default=None,
        metavar="PATH",
        help=(
            "Optional uninformed-model .ckpt (must be the same dataset and variables as "
            "the main OAT checkpoint). If given, two extra figures are produced: "
            "ERA5 | Uninformed | COSMO fields, and OAT-Uninformed / OAT-COSMO differences."
        ),
    )
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
    return p.parse_args()


def _predict_grid_mean(plotter, snap_idx, query_coords, var_idx, batch_size, grid_shape, logger):
    """Batched grid inference → predictive-mean field reshaped to *grid_shape*."""
    means = []
    N = len(query_coords)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        samples = plotter._run_inference(snap_idx, query_coords[start:end])  # [B, n_samples, n_vars]
        means.append(samples[:, :, var_idx].mean(dim=1).detach().cpu().numpy())
        logger.info("Inference: %d / %d", end, N)
    return np.concatenate(means).reshape(grid_shape)


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

    # Batched inference (OAT)
    mean_field = _predict_grid_mean(
        plotter, snap_idx, query_coords, var_idx, args.batch_size, grid_shape, logger
    )
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

    # The 3x3 comparison figure requires the uninformed model and eval stations.
    if args.uninformed_checkpoint is None:
        logger.error("--uninformed-checkpoint is required for the 3x3 comparison figure.")
        sys.exit(1)
    if eval_indices is None:
        logger.error("--split {val,test} is required (row 3 shows errors at eval stations).")
        sys.exit(1)

    # Run the uninformed model on the same grid/time (same dataset & vars).
    logger.info("Loading uninformed checkpoint %s", args.uninformed_checkpoint)
    u_pred, u_cfg = load_checkpoint(OATPredictor, args.uninformed_checkpoint, map_location="cpu")
    u_cfg.dataset.data_points_for_stats = 0
    u_pred = u_pred.eval()
    if var not in u_pred.var_to_indx:
        logger.error(
            "Variable '%s' not in uninformed checkpoint. Available: %s",
            var, list(u_pred.var_to_indx),
        )
        sys.exit(1)
    u_var_idx = u_pred.var_to_indx[var]
    u_dataset, *_ = setup_data_and_transform(u_cfg, logger)
    u_snap_idx = _resolve_time_idx(u_dataset, snap_time_iso)
    u_plotter = OATPlotter(
        predictor=u_pred, dataset=u_dataset, n_samples=args.n_samples,
        device="cpu", plot_config=_load_plot_config(u_dataset),
    )
    uninf_field = _predict_grid_mean(
        u_plotter, u_snap_idx, query_coords, u_var_idx, args.batch_size, grid_shape, logger,
    )

    import matplotlib.colors as mcolors
    var_cfg = plotter._get_var_plot_config(var)
    cmap = var_cfg.get("cmap", "viridis")
    ckpt_tag = Path(args.checkpoint).parent.name

    # Shared colour range across every absolute field (OAT, Uninformed, COSMO, ERA5) so
    # the OAT and Uninformed figures are directly comparable.
    all_fields = [mean_field]
    if uninf_field is not None:
        all_fields.append(uninf_field)
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

    if era5_field_2d is None or cosmo_field is None:
        logger.error("Both --era5-var and --nwp-var are required for the 3x3 figure.")
        sys.exit(1)

    proj = ccrs.PlateCarree()
    mean_lat = 0.5 * (lat_min + lat_max)

    # ---- Row 2: field differences vs COSMO (diverging, symmetric about 0) ----
    diff_cu = cosmo_field - uninf_field
    diff_co = cosmo_field - mean_field
    dvals = np.concatenate([diff_cu.ravel(), diff_co.ravel()])
    dmax = float(np.nanpercentile(np.abs(dvals), 99))
    if dmax < 1e-6:
        dmax = 1.0
    diff_norm = mcolors.Normalize(vmin=-dmax, vmax=dmax)

    # ---- Row 3: prediction − obs at eval (test) stations ----
    from scipy.spatial import cKDTree
    eval_coords = dataset.station_coords[eval_indices].to(torch.float32)  # [E, 2] (lon, lat)
    obs_col = dataset.obs_tensor[snap_idx, eval_indices, var_idx].numpy()
    obs_valid = dataset.obs_mask_tensor[snap_idx, eval_indices, var_idx].numpy().astype(bool)
    obs_valid &= np.isfinite(obs_col)

    # Model predictions at the exact station coords (test stations are held out of the
    # OAT conditioning set, so this is a valid eval).
    oat_at_st = _predict_grid_mean(
        plotter, snap_idx, eval_coords, var_idx, args.batch_size, (len(eval_indices),), logger
    )
    uninf_at_st = _predict_grid_mean(
        u_plotter, u_snap_idx, eval_coords, u_var_idx, args.batch_size, (len(eval_indices),), logger
    )
    tree = cKDTree(np.column_stack([lon_grid.ravel(), lat_grid.ravel()]))
    _, nn = tree.query(eval_coords.numpy())
    cosmo_at_st = cosmo_field.ravel()[nn]

    st_lon = eval_coords[:, 0].numpy()
    st_lat = eval_coords[:, 1].numpy()
    # Keep only stations inside the actual COSMO grid footprint — a rotated quadrilateral
    # with curved lon/lat edges, not a lon/lat box. Otherwise the nearest-cell COSMO lookup
    # assigns far-away boundary values to out-of-domain stations, unfairly inflating COSMO's
    # error (and skewing the shared colour range). The outline traces every node along the
    # outer edge of the structured grid, so its curvature is followed to grid resolution.
    from matplotlib.path import Path as _MplPath
    cosmo_boundary = np.vstack([
        np.column_stack([lon_grid[0, :], lat_grid[0, :]]),         # top edge
        np.column_stack([lon_grid[:, -1], lat_grid[:, -1]]),       # right edge
        np.column_stack([lon_grid[-1, ::-1], lat_grid[-1, ::-1]]), # bottom edge (reversed)
        np.column_stack([lon_grid[::-1, 0], lat_grid[::-1, 0]]),   # left edge (reversed)
    ])
    in_domain = _MplPath(cosmo_boundary).contains_points(np.column_stack([st_lon, st_lat]))
    obs_valid &= in_domain
    station_err = {
        "COSMO": cosmo_at_st - obs_col,
        "Uninformed": uninf_at_st - obs_col,
        "OAT": oat_at_st - obs_col,
    }
    all_err = np.concatenate([station_err[k][obs_valid] for k in station_err])
    emax = float(np.nanpercentile(np.abs(all_err), 99)) if all_err.size else 1.0
    if emax < 1e-6:
        emax = 1.0
    err_norm = mcolors.Normalize(vmin=-emax, vmax=emax)
    logger.info(
        "Diff range ±%.3g | error range ±%.3g | %d valid test stations",
        dmax, emax, int(obs_valid.sum()),
    )

    # ---- 3x3 figure ----
    diff_cmap = "PuOr"
    # Fixed width → predictable output size and consistent fonts across figures. Height is
    # derived so the cos(lat)-aspect maps fill their cells (knobs in the config block up top).
    map_wh = (lon_max - lon_min) * np.cos(np.deg2rad(mean_lat)) / (lat_max - lat_min)
    map_w_in = (FIG_WIDTH_IN - _MARGIN_W_IN) / 3   # 3 columns
    map_h_in = map_w_in / map_wh                   # keep the geographic aspect
    fig_h = 3 * map_h_in + _MARGIN_H_IN            # 3 rows
    fig, axs = plt.subplots(
        3, 3, figsize=(FIG_WIDTH_IN, fig_h), layout="constrained",
        subplot_kw={"projection": proj},
    )
    # Trim inter-row padding so rows sit close.
    fig.get_layout_engine().set(h_pad=0.03, hspace=0.0)

    # Row 0: absolute fields (sequential, shared colorbar)
    seq_mesh = axs[0, 0].pcolormesh(
        era5_lon_grid, era5_lat_grid, era5_field_2d, cmap=cmap, norm=shared_norm,
        shading="auto", transform=proj, rasterized=True,
    )
    axs[0, 0].set_title(f"ERA5 {args.era5_var}")
    axs[0, 1].pcolormesh(
        lon_grid, lat_grid, uninf_field, cmap=cmap, norm=shared_norm,
        shading="auto", transform=proj, rasterized=True,
    )
    axs[0, 1].set_title("Uninformed")
    axs[0, 2].pcolormesh(
        lon_grid, lat_grid, mean_field, cmap=cmap, norm=shared_norm,
        shading="auto", transform=proj, rasterized=True,
    )
    axs[0, 2].set_title("OAT")

    # Row 1: COSMO (sequential) + differences vs COSMO (diverging, shared)
    axs[1, 0].pcolormesh(
        lon_grid, lat_grid, cosmo_field, cmap=cmap, norm=shared_norm,
        shading="auto", transform=proj, rasterized=True,
    )
    axs[1, 0].set_title(f"COSMO {args.nwp_var}")
    diff_mesh = axs[1, 1].pcolormesh(
        lon_grid, lat_grid, diff_cu, cmap=diff_cmap, norm=diff_norm,
        shading="auto", transform=proj, rasterized=True,
    )
    axs[1, 1].set_title("Δ = COSMO − Uninformed")
    axs[1, 2].pcolormesh(
        lon_grid, lat_grid, diff_co, cmap=diff_cmap, norm=diff_norm,
        shading="auto", transform=proj, rasterized=True,
    )
    axs[1, 2].set_title("Δ = COSMO − OAT")

    # Row 2: base map + eval-station error markers (diverging, shared)
    err_sc = None
    for col, key in enumerate(("COSMO", "Uninformed", "OAT")):
        sc = axs[2, col].scatter(
            st_lon[obs_valid], st_lat[obs_valid], c=station_err[key][obs_valid],
            cmap=diff_cmap, norm=err_norm, marker="s", s=25,
            edgecolor="k", linewidth=0.3, transform=proj, zorder=5,
        )
        axs[2, col].plot(
            cosmo_boundary[:, 0], cosmo_boundary[:, 1], color="0.15",
            linewidth=1.5, transform=proj, zorder=4,
        )
        axs[2, col].set_title(f"Δ = {key} − Obs")
        mae = float(np.mean(np.abs(station_err[key][obs_valid])))
        axs[2, col].text(
            0.02, 0.98, f"MAE = {mae:.2f} {cb_label}",
            transform=axs[2, col].transAxes, ha="left", va="top", zorder=10,
            fontsize=14,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
        )
        if err_sc is None:
            err_sc = sc

    # Extent, cos(lat) aspect, gridlines, borders for every panel
    for r in range(3):
        for c in range(3):
            ax = axs[r, c]
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
            ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = c == 0
            gl.bottom_labels = r == 2
    try:
        _add_borders(list(axs.ravel()), lon_grid[0], lat_grid[:, 0])
    except Exception as exc:
        logger.warning("Geo overlays skipped: %s", exc)

    # Three colorbars, one per quantity (row). extend="both" draws triangular ends to
    # signal the limits are percentiles (not true min/max) — data extends beyond them.
    fig.colorbar(
        seq_mesh, ax=list(axs[0, :]), location="right", shrink=0.9,
        label=cb_label, extend="both",
    )
    fig.colorbar(
        diff_mesh, ax=list(axs[1, :]), location="right", shrink=0.9,
        label=f"Δ {cb_label}", extend="both",
    )
    fig.colorbar(
        err_sc, ax=list(axs[2, :]), location="right", shrink=0.9,
        label=f"Δ {cb_label}", extend="both",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_stem = f"{ckpt_tag}_{var}_{snap_time_iso}_comparison_3x3"
    for ext in ("png", "pdf"):
        out_path = OUT_DIR / f"{out_stem}.{ext}"
        # No bbox_inches="tight": it would crop to content and make the saved size
        # unpredictable. constrained_layout already fits everything inside FIG_WIDTH_IN.
        fig.savefig(out_path, dpi=300)
        logger.info("Saved → %s", out_path)
    plt.close(fig)


if __name__ == "__main__":
    print("Running OAT NWP-grid diagnostic...")
    main()
