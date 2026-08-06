"""Publication figure: NWP interp vs pre-OAT vs OAT, plus assimilation delta.

A 2x4 grid. Rows are variables. Columns:
  1. NWP field  (NWPInterpolatorPredictor)
  2. Pre-OAT prediction  (OAT model output before the assimilation block)
  3. OAT prediction  (full model)
  4. pre_oat - oat difference field; stations coloured by (pre_oat - obs).

Uses one OAT checkpoint (columns 2, 3, and the column-4 station residuals) plus
the NWP interpolator.

Rows plot the predictive mean by default.  With ``--prob-above VAR=THRESH`` the
row for VAR instead plots the exceedance probability P(VAR > THRESH), which is
the more meaningful summary for precipitation (whose predictive distribution has
a point mass at zero, so its mean is a poor description of the field).

Usage
-----
    python -m scripts.plot_assimilation_panel --dataset cosmo_peakweather \\
        --time 2021-08-04:00 --n-samples 100
    python -m scripts.plot_assimilation_panel --center 2660000 1190000 \\
        --extent 40000
    python -m scripts.plot_assimilation_panel --prob-above precipitation=1.0
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
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
    p.add_argument(
        "--prob-above",
        nargs="+",
        default=None,
        metavar="VAR=THRESH",
        help="Plot P(VAR > THRESH) instead of the mean for these variables, "
        "with THRESH in the variable's physical units "
        "(e.g. --prob-above precipitation=1.0).",
    )
    p.add_argument("--out", default=None)
    return p.parse_args()


def _parse_prob_above(items: list[str] | None) -> dict[str, float]:
    """Parse ``VAR=THRESH`` pairs into {variable: threshold}."""
    if not items:
        return {}
    out: dict[str, float] = {}
    for item in items:
        var, _, raw = item.partition("=")
        if not var or not raw:
            raise ValueError(
                f"--prob-above expects VAR=THRESH pairs, got {item!r}."
            )
        out[var] = float(raw)
    return out


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
    prob_above = _parse_prob_above(args.prob_above)
    unknown = set(prob_above) - set(variables)
    if unknown:
        raise ValueError(
            f"--prob-above names variables that are not plotted: {sorted(unknown)}. "
            f"Available: {variables}."
        )
    logger.info("Variables (rows): %s at %s", variables, time_iso)
    for var, thresh in prob_above.items():
        logger.info("%s: plotting P(> %g) instead of the mean", var, thresh)

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

    def _prob_field(var, threshold, predictor=None, stage="oat"):
        return plotter.predict_exceedance_prob(
            grid_coords,
            time_idx,
            var,
            threshold,
            predictor=predictor,
            stage=stage,
        ).reshape(y_grid.shape)

    n_rows = len(variables)
    # constrained_layout packs the equal-aspect panels and colorbars tightly,
    # removing the large gaps a plain grid leaves around wide-and-short maps.
    fig, axes = plt.subplots(
        n_rows, 4, figsize=(20, 3.4 * n_rows), squeeze=False, layout="constrained"
    )

    # (label, colour) of the exceedance ring, one per probability row, for the
    # figure legend built after the loop.
    ring_specs: list[tuple[str, Any]] = []

    for row, var in enumerate(variables):
        var_idx = predictor_oat.var_to_indx[var]
        var_cfg = (
            plot_config[var] if (plot_config is not None and var in plot_config) else {}
        )
        amount_cmap = var_cfg.get("cmap", "viridis")
        threshold = prob_above.get(var)
        # Probability rows use a colormap distinct from the amount colormap (and
        # from the PuOr difference panel), so the field, the station fills and
        # the difference panel are never confused for one another. Its top colour
        # doubles as the "observed the event" marker edge, so it must stay
        # visible on white and on a filled marker.
        cmap = var_cfg.get("prob_cmap", "Greens") if threshold is not None else amount_cmap

        if threshold is None:
            nwp_field = _mean_field(var_idx, predictor=nwp_predictor, stage="oat")
            pre_field = _mean_field(var_idx, stage="pre_oat")
            oat_field = _mean_field(var_idx, stage="oat")
        else:
            # The NWP interpolator is deterministic, so its "probability" is the
            # 0/1 indicator of its point forecast exceeding the threshold.
            nwp_field = _prob_field(var, threshold, predictor=nwp_predictor)
            pre_field = _prob_field(var, threshold, stage="pre_oat")
            oat_field = _prob_field(var, threshold, stage="oat")

        unit = units.get(var, "")
        var_label = var.capitalize()
        if threshold is None:
            quantity_label = f"{var_label} [{unit}]" if unit else var_label
        else:
            thresh_label = f"{threshold:g} {unit}".strip()
            quantity_label = f"P({var_label} > {thresh_label})"

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
                vmin=var_cfg.get("vmin") if threshold is None else 0.0,
                vmax=var_cfg.get("vmax") if threshold is None else 1.0,
            )
            # Rasterize the dense field so PDF output stays small and renders
            # cleanly; markers, borders and text remain vector.
            mesh.set_rasterized(True)
            meshes.append(mesh)
            if row == 0:
                axes[row, col].set_title(name, fontsize=13)
        # Probabilities are already on a fixed, comparable [0, 1] scale; only the
        # physical fields need their shared limits harmonized after the fact.
        if threshold is None:
            vmin = float(
                np.nanmin([np.nanmin(f) for f in (nwp_field, pre_field, oat_field)])
            )
            vmax = float(
                np.nanmax([np.nanmax(f) for f in (nwp_field, pre_field, oat_field)])
            )
        # On a mean row the markers share the field's scale, so the field mesh
        # supplies their colours. On a probability row the field is a probability
        # but the observation is an amount, so the markers get their own amount
        # colormap (with its own colorbar) and a coloured edge — the top of the
        # probability colormap — marks the stations that observed the event.
        station_mesh: list[Any] = list(meshes)
        edge_color_by: dict[int, Any] | None = None
        size_scale = 1.0
        amount_sm = None
        if threshold is not None:
            observed = {
                s: dataset.obs_tensor[time_idx, s, var_idx].item()
                for s in sorted(set(obs_station_indices) | set(eval_indices))
                if dataset.obs_mask_tensor[time_idx, s, var_idx].item()
            }
            # Observed amounts are heavily right-skewed, so scaling up to the
            # maximum lets a single wet station compress every other marker into
            # the first few percent of the colormap. Mean rows avoid this only by
            # accident: they inherit the (much narrower) predictive-mean field
            # range and clip whatever exceeds it. There is no amount field to
            # inherit from here, so clip explicitly at a robust upper limit,
            # never below the threshold so it stays on the scale. The colorbar is
            # drawn with extend="max" to show that the top end is clipped.
            amounts = np.array(list(observed.values()))
            positive = amounts[amounts > 0]
            amount_max = max(
                threshold,
                float(np.quantile(positive, 0.98)) if positive.size else threshold,
            )
            amount_sm = ScalarMappable(
                norm=Normalize(vmin=var_cfg.get("vmin", 0.0), vmax=amount_max),
                cmap=amount_cmap,
            )
            station_mesh = [amount_sm] * 3
            exceed_color = plt.get_cmap(cmap)(1.0)
            edge_color_by = {
                s: exceed_color for s, v in observed.items() if v >= threshold
            }
            # Both the fill and the ring have to stay readable on one marker.
            size_scale = 2.5
            ring_specs.append((f"Observed {var} ≥ {thresh_label}", exceed_color))

        for col in range(3):
            if threshold is None:
                meshes[col].set_clim(vmin, vmax)
            plotter.overlay_stations(
                axes[row, col],
                time_idx,
                var_idx,
                mesh=station_mesh[col],
                edge_color_by=edge_color_by,
                size_scale=size_scale,
            )
        axes[row, 0].set_ylabel(quantity_label, fontsize=13)
        fig.colorbar(
            meshes[-1], ax=list(axes[row, :3]), label=quantity_label, shrink=0.85
        )
        if amount_sm is not None:
            fig.colorbar(
                amount_sm,
                ax=list(axes[row, :3]),
                label=f"Observed {var} [{unit}]" if unit else f"Observed {var}",
                shrink=0.85,
                extend="max",
            )

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
            if threshold is None:
                pre_at = (
                    plotter.predict_samples(st_xy, time_idx, stage="pre_oat")[
                        :, :, var_idx
                    ]
                    .mean(dim=1)
                    .numpy()
                )
            else:
                pre_at = plotter.predict_exceedance_prob(
                    st_xy, time_idx, var, threshold, stage="pre_oat"
                )
            for i, s in enumerate(colored):
                obs_v = dataset.obs_tensor[time_idx, s, var_idx].item()
                if threshold is not None:
                    # Brier-style residual: predicted probability − observed
                    # 0/1 outcome, matching the units of the Δ-probability field.
                    obs_v = float(obs_v > threshold)
                color_by[s] = float(pre_at[i] - obs_v)
        # Here the fill has to stay the residual (that is the panel's point), but
        # the exceedance rings carry over so the same stations are identifiable
        # across all four columns.
        plotter.overlay_stations(
            axes[row, 3],
            time_idx,
            var_idx,
            mesh=mesh_d,
            color_by=color_by,
            edge_color_by=edge_color_by,
            size_scale=size_scale,
        )
        if row == 0:
            axes[row, 3].set_title("Pre-OAT − OAT", fontsize=13)  # noqa: RUF001
        dlabel = f"Δ {quantity_label}"
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
    station_handles += [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.6,
            markersize=8,
            label=label,
        )
        for label, color in ring_specs
    ]
    fig.legend(
        handles=station_handles,
        loc="outside lower center",
        ncol=min(4, len(station_handles)),
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
    # Tag probability figures so they do not overwrite the mean-field figure for
    # the same dataset and timestep.
    prob_tag = "".join(f"_p-{v}-gt-{t:g}" for v, t in sorted(prob_above.items()))
    out_path = (
        Path(args.out)
        if args.out is not None
        else OUT_DIR / f"assimilation_panel_{args.dataset}_{time_iso}{prob_tag}.png"
    )
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
