"""Publication figure: NWP interp vs pre-OAT vs OAT, plus assimilation delta.

A 2x4 grid. Rows are variables. Columns:
  1. NWP field  (NWPInterpolatorPredictor)
  2. Pre-OAT prediction  (OAT model output before the assimilation block)
  3. OAT prediction  (full model)
  4. pre_oat - oat difference field; stations coloured by (pre_oat - obs).

Uses one OAT checkpoint (columns 2, 3, and the column-4 station residuals) plus
the NWP interpolator.

Usage
-----
    python -m scripts.plot_assimilation_panel --dataset cosmo_peakweather \\
        --time 2021-08-04:00 --n-samples 100
    python -m scripts.plot_assimilation_panel --center 2660000 1190000 \\
        --extent 40000
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from omegaconf import OmegaConf

from oat.data_module.cosmo_peakweather_dataset import CosmoPeakWeatherDataset
from oat.data_module.era5_meteostat_dataset import ERA5MeteostatDataset
from oat.data_module.setup import setup_data_and_transform
from oat.model_module.baseline import NWPInterpolatorConfig
from oat.plotting_module import OATPlotter
from oat.plotting_module.geo_overlays import _add_borders, build_domain_grid
from oat.predictor_module.baseline_predictor import NWPInterpolatorPredictor
from oat.predictor_module.checkpoint_utils import load_checkpoint
from oat.predictor_module.oat_predictor import OATPredictor

_PROJECT_ROOT = Path(__file__).parents[1]
OUT_DIR = _PROJECT_ROOT / "data/figures"
_CHECKPOINTS_YAML = _PROJECT_ROOT / "configs/checkpoints.yaml"
_PLOT_CONFIG_PATHS = {
    ERA5MeteostatDataset: _PROJECT_ROOT
    / "configs/dataset/era5_meteostat/plot_config.yaml",
    CosmoPeakWeatherDataset: _PROJECT_ROOT
    / "configs/dataset/cosmo_peakweather/plot_config.yaml",
}


def _load_plot_config(dataset):
    path = _PLOT_CONFIG_PATHS.get(type(dataset))
    if path and path.exists():
        return OmegaConf.load(path).obs_plot_config
    return None


def _load_units(dataset_name: str) -> dict[str, str]:
    """Map observed-variable name -> unit string from the dataset catalog."""
    path = _PROJECT_ROOT / f"configs/dataset/{dataset_name}/catalog.yaml"
    if not path.exists():
        return {}
    obs = OmegaConf.load(path).get("obs_vars", {}) or {}
    return {v: str(obs[v].get("unit", "")) for v in obs}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="cosmo_peakweather")
    p.add_argument("--time", default=None, metavar="YYYY-MM-DD[:HH]")
    p.add_argument("--variables", nargs="+", default=None)
    p.add_argument("--center", nargs=2, type=float, default=None, metavar=("E", "N"))
    p.add_argument("--extent", type=float, default=None)
    p.add_argument("--resolution", type=float, default=None)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--out", default=None)
    return p.parse_args()


def _resolve_time_idx(dataset, time_str: str) -> int:
    ts = pd.Timestamp(time_str.replace(":", " "), tz="UTC")
    idx = int(dataset.times.searchsorted(ts))
    return min(idx, len(dataset) - 1)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)
    args = _parse_args()

    registry = OmegaConf.load(_CHECKPOINTS_YAML)
    oat_ckpt = registry[args.dataset]["oat_temp_precip"]
    predictor_oat, cfg = load_checkpoint(OATPredictor, oat_ckpt, map_location="cpu")
    predictor_oat = predictor_oat.eval()

    cfg.dataset.data_points_for_stats = 0
    dataset, _, _, _val_loader, test_loader, *_ = setup_data_and_transform(cfg, logger)

    test_ds = test_loader.dataset
    obs_station_indices = list(test_ds.observed_indices)
    eval_indices = list(test_ds.eval_indices)
    logger.info(
        "Test split: %d observed, %d eval stations",
        len(obs_station_indices),
        len(eval_indices),
    )

    nwp_predictor = NWPInterpolatorPredictor(
        config=NWPInterpolatorConfig(field_extent=dataset.nwp_extent),
        output_var_names=dataset.config.obs_vars,
        input_transform=dataset.get_baseline_pre_transform(),
    ).eval()

    if args.time is not None:
        time_idx = _resolve_time_idx(dataset, args.time)
    elif test_ds._time_indices is not None and len(test_ds._time_indices):
        # Default to the first held-out test timestep, not dataset index 0
        # (which falls in the train period).
        time_idx = int(test_ds._time_indices[0])
    else:
        time_idx = 0
    time_iso = dataset.times[time_idx].strftime("%Y-%m-%dT%H")

    variables = args.variables or list(predictor_oat.var_to_indx.keys())
    plot_config = _load_plot_config(dataset)
    units = _load_units(args.dataset)
    logger.info("Variables (rows): %s at %s", variables, time_iso)

    plotter = OATPlotter(
        predictor=predictor_oat,
        dataset=dataset,
        obs_station_indices=obs_station_indices,
        eval_indices=eval_indices,
        n_samples=args.n_samples,
        device="cpu",
        plot_config=plot_config,
    )

    center = tuple(args.center) if args.center is not None else None
    xs, ys = build_domain_grid(
        dataset, resolution=args.resolution, center=center, extent=args.extent
    )
    if args.resolution is None:
        # build_domain_grid's default (~100 cells across the width) is too coarse
        # for a publication figure; refine to ~250 cells across the same extent.
        fine_res = (xs[-1] - xs[0]) / 250.0
        xs, ys = build_domain_grid(
            dataset, resolution=fine_res, center=center, extent=args.extent
        )
    x_grid, y_grid = np.meshgrid(xs, ys)
    grid_coords = torch.tensor(
        np.stack([x_grid.ravel(), y_grid.ravel()], axis=1), dtype=torch.float32
    )
    logger.info("Grid %dx%d (%d points)", len(xs), len(ys), len(xs) * len(ys))

    def _mean_field(var_idx, predictor=None, stage="oat"):
        samples = plotter.predict_samples(
            grid_coords, time_idx, predictor=predictor, stage=stage
        )  # [N_q, n_samples, n_vars]
        return (
            samples[:, :, var_idx]
            .mean(dim=1)
            .detach()
            .cpu()
            .numpy()
            .reshape(y_grid.shape)
        )

    n_rows = len(variables)
    # constrained_layout packs the equal-aspect panels and colorbars tightly,
    # removing the large gaps a plain grid leaves around wide-and-short maps.
    fig, axes = plt.subplots(
        n_rows, 4, figsize=(20, 3.4 * n_rows), squeeze=False, layout="constrained"
    )

    for row, var in enumerate(variables):
        var_idx = predictor_oat.var_to_indx[var]
        var_cfg = (
            plot_config[var] if (plot_config is not None and var in plot_config) else {}
        )
        cmap = var_cfg.get("cmap", "viridis")

        nwp_field = _mean_field(var_idx, predictor=nwp_predictor, stage="oat")
        pre_field = _mean_field(var_idx, stage="pre_oat")
        oat_field = _mean_field(var_idx, stage="oat")

        unit = units.get(var, "")
        var_label = var.capitalize()

        # Columns 1-3: shared colormap + harmonized limits.
        panels = [
            ("NWP interpolation", nwp_field),
            ("Pre-OAT", pre_field),
            ("OAT", oat_field),
        ]
        meshes = []
        for col, (name, fld) in enumerate(panels):
            mesh = plotter.draw_field(
                axes[row, col],
                fld,
                xs,
                ys,
                cmap=cmap,
                vmin=var_cfg.get("vmin"),
                vmax=var_cfg.get("vmax"),
            )
            # Rasterize the dense field so PDF output stays small and renders
            # cleanly; markers, borders and text remain vector.
            mesh.set_rasterized(True)
            meshes.append(mesh)
            if row == 0:
                axes[row, col].set_title(name, fontsize=13)
        vmin = float(
            np.nanmin([np.nanmin(f) for f in (nwp_field, pre_field, oat_field)])
        )
        vmax = float(
            np.nanmax([np.nanmax(f) for f in (nwp_field, pre_field, oat_field)])
        )
        for col in range(3):
            meshes[col].set_clim(vmin, vmax)
            plotter.overlay_stations(
                axes[row, col], time_idx, var_idx, mesh=meshes[col]
            )
        axes[row, 0].set_ylabel(
            f"{var_label} [{unit}]" if unit else var_label, fontsize=13
        )
        clabel = f"{var_label} [{unit}]" if unit else var_label
        fig.colorbar(meshes[-1], ax=list(axes[row, :3]), label=clabel, shrink=0.85)

        # Column 4: pre_oat - oat difference; stations coloured by pre_oat - obs.
        # Use a diverging colormap distinct from the field cmaps (temperature is
        # RdBu_r, precipitation is Blues) so the difference panels are not
        # confused with the prediction panels.
        diff = pre_field - oat_field
        vmax_d = float(np.nanmax(np.abs(diff))) or 1.0
        mesh_d = plotter.draw_field(
            axes[row, 3], diff, xs, ys, cmap="PuOr", vmin=-vmax_d, vmax=vmax_d
        )
        mesh_d.set_rasterized(True)
        colored = [
            s
            for s in sorted(set(obs_station_indices) | set(eval_indices))
            if dataset.obs_mask_tensor[time_idx, s, var_idx].item()
        ]
        color_by: dict[int, float] = {}
        if colored:
            st_xy = torch.tensor(
                [
                    [
                        dataset.station_coords[s, 0].item(),
                        dataset.station_coords[s, 1].item(),
                    ]
                    for s in colored
                ],
                dtype=torch.float32,
            )
            pre_at = (
                plotter.predict_samples(st_xy, time_idx, stage="pre_oat")[:, :, var_idx]
                .mean(dim=1)
                .numpy()
            )
            for i, s in enumerate(colored):
                obs_v = dataset.obs_tensor[time_idx, s, var_idx].item()
                color_by[s] = float(pre_at[i] - obs_v)
        plotter.overlay_stations(
            axes[row, 3], time_idx, var_idx, mesh=mesh_d, color_by=color_by
        )
        if row == 0:
            axes[row, 3].set_title("Pre-OAT − OAT", fontsize=13)  # noqa: RUF001
        dlabel = f"Δ {var_label} [{unit}]" if unit else f"Δ {var_label}"
        fig.colorbar(mesh_d, ax=axes[row, 3], label=dlabel, shrink=0.85)

    # Drop the grey "missing" evaluation-station squares (test stations with no
    # observation at this timestep) for a cleaner figure. The legend uses proxy
    # handles, so removing these collections does not affect it.
    for ax in axes.ravel():
        for coll in list(ax.collections):
            if coll.get_label() == "eval missing":
                coll.remove()

    # Equal aspect so the fields are not horizontally squeezed (LV95 x/y share
    # units; degrees are shown plate-carrée). Drop coordinate ticks for clean
    # map panels — the borders/coastline convey geography. Done outside the
    # borders try so it always applies even if the geo overlay fails.
    for ax in axes.ravel():
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    nice_time = dataset.times[time_idx].strftime("%Y-%m-%d %H:%M UTC")
    fig.suptitle(nice_time, fontsize=15)

    # Shape legend for station markers (fill encodes the per-panel value/colorbar).
    station_handles = [
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
    ]
    fig.legend(
        handles=station_handles,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=12,
    )

    try:
        _add_borders(list(axes.ravel()), xs, ys)
        for ax in axes.ravel():
            ax.set_xlim(xs[0], xs[-1])
            ax.set_ylim(ys[0], ys[-1])
    except Exception as exc:
        logger.warning("Geo overlays skipped: %s", exc)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(args.out)
        if args.out is not None
        else OUT_DIR / f"assimilation_panel_{args.dataset}_{time_iso}.png"
    )
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
