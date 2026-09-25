"""Publication figure: ERA5/Uninformed/OAT vs COSMO-E1, over the holdout region.

A 3x3 grid, cropped to the western-Alps region-holdout bounding box, for one
variable at a time. Columns are (broadly) ERA5 / Uninformed / OAT; rows are:

  1. Absolute fields (sequential, one shared colour scale with row 2 col 0):
     ERA5 interpolation, Uninformed prediction, OAT prediction.
  2. COSMO-E1 interpolation (same shared scale as row 1), then the
     differences COSMO - Uninformed and COSMO - OAT (diverging, shared).
  3. No field - station scatter only, at the held-out test stations:
     (model - obs) for COSMO, Uninformed, OAT (diverging, shared), with a
     per-panel MAE annotation.

Context stations (train + val + test-context, revealed at test time) and
held-out test stations are drawn with the circle/square convention from
plot_assimilation_panel.py. COSMO-E1 comes from CosmoMeteostatDataset (the
same Meteostat station/topo universe, COSMO-E1 as the NWP source) via the
NWP-interpolation baseline predictor, not the raw COSMO-1E zarr used by the
older plot_ERA5_to_COSMO_grid.py (COSMO-1E's Switzerland-only domain is too
small for this region).

Usage
-----
    python -m scripts.plot_region_holdout_panel \\
        --dataset era5_meteostat --model oat_spatial_holdout_temp_precip \\
        --uninformed-model uninformed_spatial_holdout_temp_precip \\
        --time 2024-06-21:12 --var temp
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from omegaconf import OmegaConf

from oat.data_module.setup import setup_data_and_transform
from oat.plotting_module.geo_overlays import _add_borders
from oat.predictor_module.checkpoint_utils import load_checkpoint, resolve_checkpoint
from scripts.plotter_utils import (
    load_plot_config,
    maybe_build_nwp_baseline,
    select_plotter_cls,
)

_PROJECT_ROOT = Path(__file__).parents[1]
OUT_DIR = _PROJECT_ROOT / "data/figures"

# Western-Alps region-holdout bounding box, matching
# configs/split/spatial_region_holdout.yaml (see its header comment).
_DEFAULT_LAT_MIN = 43.8
_DEFAULT_LAT_MAX = 49.2
_DEFAULT_LON_MIN = 3.3
_DEFAULT_LON_MAX = 11.8

_COSMO_DATASET_OVERRIDE = "cosmo_meteostat/temp_precip"
_DIFF_CMAP = "PuOr"


def _load_units(dataset_name: str) -> dict[str, str]:
    """Map observed-variable name -> unit string from the dataset catalog."""
    path = _PROJECT_ROOT / f"configs/dataset/{dataset_name}/catalog.yaml"
    if not path.exists():
        return {}
    obs = OmegaConf.load(path).get("obs_vars", {}) or {}
    return {v: str(obs[v].get("unit", "")) for v in obs}


def _override_split(cfg: Any, split_config: str) -> Any:
    """Override cfg.split by composing configs/split/<split_config>.yaml."""
    from hydra import compose, initialize_config_dir

    config_dir = str(_PROJECT_ROOT / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        split_cfg = compose(config_name="config", overrides=[f"split={split_config}"])
    cfg.split = split_cfg.split
    return cfg


def _compose_cosmo_cfg(split_config: str) -> Any:
    """Compose a fresh config for the COSMO-E1 interpolation baseline.

    No trained checkpoint exists for cosmo_meteostat (it is eval-only, see
    evaluate_test_set.py's COSMO_INTERPOLATOR_MODEL), so this composes
    directly from configs/ instead of loading a checkpoint's saved config.
    """
    from hydra import compose, initialize_config_dir

    config_dir = str(_PROJECT_ROOT / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(
            config_name="config",
            overrides=[f"dataset={_COSMO_DATASET_OVERRIDE}", f"split={split_config}"],
        )


def _resolve_time_idx(dataset, time_str: str) -> int:
    """Return the index of ``time_str`` in ``dataset.times``.

    Raise if the time is not in the dataset (e.g. after the COSMO-E1 end
    or on one of its missing dates). A nearest-time fallback would plot
    a different hour than the title shows.
    """
    ts = pd.Timestamp(time_str.replace(":", " "), tz="UTC")
    if ts not in dataset.times:
        raise ValueError(
            f"{ts} is not in the dataset times "
            f"[{dataset.times[0]}, {dataset.times[-1]}] (or is a missing date)."
        )
    return int(dataset.times.get_loc(ts))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="era5_meteostat")
    p.add_argument("--model", default="oat_spatial_holdout_temp_precip")
    p.add_argument(
        "--uninformed-model", default="uninformed_spatial_holdout_temp_precip"
    )
    p.add_argument(
        "--split-config",
        default="spatial_region_holdout",
        help="configs/split/<NAME>.yaml to use for the station split.",
    )
    p.add_argument("--time", default=None, metavar="YYYY-MM-DD[:HH]")
    p.add_argument(
        "--var", default=None, help="Variable to plot (default: first in predictor)."
    )
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument(
        "--resolution",
        type=float,
        default=0.05,
        metavar="DEG",
        help="Query grid spacing in degrees (default: 0.05).",
    )
    p.add_argument("--region-lat-min", type=float, default=_DEFAULT_LAT_MIN)
    p.add_argument("--region-lat-max", type=float, default=_DEFAULT_LAT_MAX)
    p.add_argument("--region-lon-min", type=float, default=_DEFAULT_LON_MIN)
    p.add_argument("--region-lon-max", type=float, default=_DEFAULT_LON_MAX)
    p.add_argument("--out", default=None)
    return p.parse_args()


def _load_model(
    dataset_key: str, model_key: str, split_config: str, logger: logging.Logger
) -> tuple[Any, Any, Any]:
    """Load a checkpoint plus its Field-path dataset, wired for inference.

    Returns (predictor, dataset, test_dataset) — the last exposes
    ``observed_indices``/``eval_indices`` for the region-holdout split.
    """
    ckpt = resolve_checkpoint(dataset_key, model_key)
    predictor, cfg = load_checkpoint(None, ckpt, map_location="cpu")
    predictor = predictor.eval()
    cfg = _override_split(cfg, split_config)
    cfg.dataset.data_points_for_stats = 0
    dataset, _, _, _val_loader, test_loader, *_ = setup_data_and_transform(cfg, logger)
    if hasattr(predictor, "set_index_projection"):
        predictor.set_index_projection(dataset)
    return predictor, dataset, test_loader.dataset


def _load_cosmo_baseline(
    split_config: str, logger: logging.Logger
) -> tuple[Any, Any, Any]:
    """Build the COSMO-E1 interpolation baseline predictor + its dataset."""
    cfg = _compose_cosmo_cfg(split_config)
    cfg.dataset.data_points_for_stats = 0
    dataset, _, _, _val_loader, test_loader, *_ = setup_data_and_transform(cfg, logger)
    predictor = maybe_build_nwp_baseline(dataset, logger, required=True)
    return predictor, dataset, test_loader.dataset


def _check_var(predictor: Any, var: str, tag: str, logger: logging.Logger) -> None:
    if var not in predictor.var_to_indx:
        logger.error(
            "Variable '%s' not in %s. Available: %s",
            var,
            tag,
            list(predictor.var_to_indx),
        )
        sys.exit(1)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)
    args = _parse_args()

    predictor_oat, dataset, test_ds = _load_model(
        args.dataset, args.model, args.split_config, logger
    )
    observed = list(map(int, test_ds.observed_indices))
    held_out = list(map(int, test_ds.eval_indices))
    logger.info(
        "Region-holdout split: %d context (assimilated), %d held-out test stations",
        len(observed),
        len(held_out),
    )

    predictor_uninf, u_dataset, _u_test_ds = _load_model(
        args.dataset, args.uninformed_model, args.split_config, logger
    )
    era5_predictor = maybe_build_nwp_baseline(dataset, logger, required=True)
    cosmo_predictor, cosmo_dataset, cosmo_test_ds = _load_cosmo_baseline(
        args.split_config, logger
    )
    cosmo_observed = list(map(int, cosmo_test_ds.observed_indices))
    cosmo_held_out = list(map(int, cosmo_test_ds.eval_indices))

    var = args.var if args.var is not None else next(iter(predictor_oat.var_to_indx))
    _check_var(predictor_oat, var, "OAT checkpoint", logger)
    _check_var(predictor_uninf, var, "Uninformed checkpoint", logger)
    _check_var(cosmo_predictor, var, "COSMO baseline", logger)

    if args.time is not None:
        time_idx = _resolve_time_idx(dataset, args.time)
    elif test_ds._time_indices is not None and len(test_ds._time_indices):
        # Default to the first held-out test timestep, not dataset index 0
        # (which falls in the train period).
        time_idx = int(test_ds._time_indices[0])
    else:
        time_idx = 0
    time_iso = dataset.times[time_idx].strftime("%Y-%m-%dT%H")
    u_time_idx = _resolve_time_idx(u_dataset, time_iso)
    cosmo_time_idx = _resolve_time_idx(cosmo_dataset, time_iso)
    logger.info("Variable: %s at %s", var, time_iso)

    var_idx = predictor_oat.var_to_indx[var]
    u_var_idx = predictor_uninf.var_to_indx[var]
    cosmo_var_idx = cosmo_predictor.var_to_indx[var]

    plot_config = load_plot_config(dataset)
    units = _load_units(args.dataset)
    var_cfg = (
        plot_config[var] if (plot_config is not None and var in plot_config) else {}
    )
    cmap = var_cfg.get("cmap", "viridis")
    unit = units.get(var, "")
    var_label = var.capitalize()
    cb_label = f"{var_label} [{unit}]" if unit else var_label

    PlotterCls = select_plotter_cls(dataset)
    plotter = PlotterCls(
        predictor=predictor_oat,
        dataset=dataset,
        obs_station_indices=observed,
        eval_indices=held_out,
        n_samples=args.n_samples,
        device="cpu",
        plot_config=plot_config,
    )
    UPlotterCls = select_plotter_cls(u_dataset)
    u_plotter = UPlotterCls(
        predictor=predictor_uninf,
        dataset=u_dataset,
        n_samples=args.n_samples,
        device="cpu",
        plot_config=load_plot_config(u_dataset),
    )
    CosmoPlotterCls = select_plotter_cls(cosmo_dataset)
    cosmo_plotter = CosmoPlotterCls(
        predictor=cosmo_predictor,
        dataset=cosmo_dataset,
        obs_station_indices=cosmo_observed,
        eval_indices=cosmo_held_out,
        n_samples=args.n_samples,
        device="cpu",
        plot_config=load_plot_config(cosmo_dataset),
    )

    lon_min, lon_max = args.region_lon_min, args.region_lon_max
    lat_min, lat_max = args.region_lat_min, args.region_lat_max
    xs = np.arange(lon_min, lon_max + 1e-9, args.resolution)
    ys = np.arange(lat_min, lat_max + 1e-9, args.resolution)
    x_grid, y_grid = np.meshgrid(xs, ys)
    grid_coords = torch.tensor(
        np.stack([x_grid.ravel(), y_grid.ravel()], axis=1), dtype=torch.float32
    )
    logger.info("Grid %dx%d (%d points)", len(xs), len(ys), len(xs) * len(ys))

    def _mean_field(plotter_, t_idx, v_idx, predictor=None):
        samples = plotter_.predict_samples(grid_coords, t_idx, predictor=predictor)
        return (
            samples[:, :, v_idx]
            .mean(dim=1)
            .detach()
            .cpu()
            .numpy()
            .reshape(y_grid.shape)
        )

    def _mean_at(plotter_, coords, t_idx, v_idx, predictor=None):
        samples = plotter_.predict_samples(coords, t_idx, predictor=predictor)
        return samples[:, :, v_idx].mean(dim=1).detach().cpu().numpy()

    # ---- Row 1: absolute fields ----
    era5_field = _mean_field(plotter, time_idx, var_idx, predictor=era5_predictor)
    uninf_field = _mean_field(u_plotter, u_time_idx, u_var_idx)
    oat_field = _mean_field(plotter, time_idx, var_idx)
    cosmo_field = _mean_field(cosmo_plotter, cosmo_time_idx, cosmo_var_idx)

    cfg_vmin = var_cfg.get("vmin")
    cfg_vmax = var_cfg.get("vmax")
    all_vals = np.concatenate(
        [f.ravel() for f in (era5_field, uninf_field, oat_field, cosmo_field)]
    )
    shared_vmin = float(
        cfg_vmin if cfg_vmin is not None else np.nanpercentile(all_vals, 1.0)
    )
    shared_vmax = float(
        cfg_vmax if cfg_vmax is not None else np.nanpercentile(all_vals, 99)
    )
    if shared_vmax - shared_vmin < 1e-6:
        shared_vmax = shared_vmin + 1.0

    # ---- Row 2: COSMO field + differences vs COSMO (diverging, symmetric) ----
    diff_cu = cosmo_field - uninf_field
    diff_co = cosmo_field - oat_field
    dvals = np.concatenate([diff_cu.ravel(), diff_co.ravel()])
    dmax = float(np.nanpercentile(np.abs(dvals), 99))
    if dmax < 1e-6:
        dmax = 1.0

    # ---- Row 3: (model - obs) at held-out test stations ----
    eval_coords = dataset.station_coords[held_out].to(torch.float32)  # [S, 2]
    obs_col = dataset.obs_tensor[time_idx, held_out, var_idx].numpy()
    obs_valid = (
        dataset.obs_mask_tensor[time_idx, held_out, var_idx].numpy().astype(bool)
    )
    obs_valid &= np.isfinite(obs_col)

    cosmo_at_st = _mean_at(cosmo_plotter, eval_coords, cosmo_time_idx, cosmo_var_idx)
    uninf_at_st = _mean_at(u_plotter, eval_coords, u_time_idx, u_var_idx)
    oat_at_st = _mean_at(plotter, eval_coords, time_idx, var_idx)
    obs_valid &= (
        np.isfinite(cosmo_at_st) & np.isfinite(uninf_at_st) & np.isfinite(oat_at_st)
    )

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
        "Diff range ±%.3g | error range ±%.3g | %d/%d valid held-out stations",
        dmax,
        emax,
        int(obs_valid.sum()),
        len(held_out),
    )

    # ---- Figure ----
    fig, axes = plt.subplots(
        3, 3, figsize=(15, 13), squeeze=False, layout="constrained"
    )

    # Row 1: ERA5, Uninformed, OAT.
    row1 = [("ERA5", era5_field), ("Uninformed", uninf_field), ("OAT", oat_field)]
    seq_meshes = []
    for col, (name, fld) in enumerate(row1):
        mesh = plotter.draw_field(
            axes[0, col], fld, xs, ys, cmap=cmap, vmin=shared_vmin, vmax=shared_vmax
        )
        mesh.set_rasterized(True)
        axes[0, col].set_title(name, fontsize=13)
        plotter.overlay_stations(axes[0, col], time_idx, var_idx, mesh=mesh)
        seq_meshes.append(mesh)

    # Row 2: COSMO (same shared scale) + differences vs COSMO.
    mesh_cosmo = cosmo_plotter.draw_field(
        axes[1, 0], cosmo_field, xs, ys, cmap=cmap, vmin=shared_vmin, vmax=shared_vmax
    )
    mesh_cosmo.set_rasterized(True)
    axes[1, 0].set_title("COSMO-E1")
    cosmo_plotter.overlay_stations(
        axes[1, 0], cosmo_time_idx, cosmo_var_idx, mesh=mesh_cosmo
    )

    diff_mesh = None
    diff_panels = (
        (1, ("Δ = COSMO − Uninformed", diff_cu)),  # noqa: RUF001
        (2, ("Δ = COSMO − OAT", diff_co)),  # noqa: RUF001
    )
    for col, (name, fld) in diff_panels:
        mesh = plotter.draw_field(
            axes[1, col], fld, xs, ys, cmap=_DIFF_CMAP, vmin=-dmax, vmax=dmax
        )
        mesh.set_rasterized(True)
        axes[1, col].set_title(name)
        # Raw obs values live on a sequential scale; colouring them on the
        # diverging diff scale would misrepresent them, so only the
        # context/held-out shapes are shown here (no colour).
        plotter.overlay_stations(axes[1, col], time_idx, var_idx)
        diff_mesh = mesh

    # Row 3: no field, just held-out-station residual scatter + borders.
    st_lon = eval_coords[:, 0].numpy()
    st_lat = eval_coords[:, 1].numpy()
    err_sc = None
    for col, key in enumerate(("COSMO", "Uninformed", "OAT")):
        sc = axes[2, col].scatter(
            st_lon[obs_valid],
            st_lat[obs_valid],
            c=station_err[key][obs_valid],
            cmap=_DIFF_CMAP,
            norm=err_norm,
            marker="s",
            s=25,
            edgecolor="k",
            linewidth=0.3,
            zorder=5,
        )
        axes[2, col].set_title(f"Δ = {key} − Obs")  # noqa: RUF001
        if err_sc is None:
            err_sc = sc

    # Drop the grey "missing" station markers (no observation at this
    # timestep) for a cleaner figure, as in plot_assimilation_panel.py.
    for ax in axes.ravel():
        for coll in list(ax.collections):
            if coll.get_label() in ("missing", "eval missing"):
                coll.remove()

    # cos(lat) aspect so the region is not distorted; degree ticks convey
    # little on a cropped panel, so drop them (borders/coastline + legend
    # carry the geography).
    mean_lat = 0.5 * (lat_min + lat_max)
    for ax in axes.ravel():
        ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(
        seq_meshes[-1], ax=list(axes[0, :]), label=cb_label, shrink=0.85, extend="both"
    )
    fig.colorbar(
        diff_mesh,
        ax=list(axes[1, :]),
        label=f"Δ {cb_label}",
        shrink=0.85,
        extend="both",
    )
    fig.colorbar(
        err_sc,
        ax=list(axes[2, :]),
        label=f"Δ {cb_label}",
        shrink=0.85,
        extend="both",
    )

    station_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=8,
            label="Context-test station",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=8,
            label="Held-out test station",
        ),
    ]
    fig.legend(
        handles=station_handles,
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize=12,
    )

    try:
        _add_borders(list(axes.ravel()), xs, ys)
        for ax in axes.ravel():
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)
    except Exception as exc:
        logger.warning("Geo overlays skipped: %s", exc)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_tag = Path(resolve_checkpoint(args.dataset, args.model)).parent.name
    out_path = (
        Path(args.out)
        if args.out is not None
        else OUT_DIR / f"region_holdout_panel_{ckpt_tag}_{var}_{time_iso}.pdf"
    )
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
