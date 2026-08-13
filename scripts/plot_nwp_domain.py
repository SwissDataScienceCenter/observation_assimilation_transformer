"""Whole-NWP-domain diagnostic plot for OAT.

Plots four spatial panels covering the full NWP input extent:
  1. NWP interpolation baseline
  2. OAT predicted mean
  3. OAT predictive std
  4. OAT − NWP difference

Usage
-----
Info mode (prints time range, domain extent, variables):
    python -m scripts.plot_nwp_domain <checkpoint>

Plot mode:
    python -m scripts.plot_nwp_domain <checkpoint> --time 2020-06-15:06 \\
        --var temperature --resolution 5000 --n-samples 50
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oat.data_module.setup import setup_data_and_transform
from oat.predictor_module.checkpoint_utils import load_checkpoint
from oat.predictor_module.oat_predictor import OATPredictor
from scripts.plotter_utils import (
    is_field_path,
    load_plot_config,
    maybe_build_nwp_baseline,
    select_plotter_cls,
)

_PROJECT_ROOT = Path(__file__).parents[1]

OUT_DIR = Path(__file__).parents[1] / "data/figures"

from oat.plotting_module.geo_overlays import (  # noqa: E402
    _add_borders,
    build_domain_grid,
    build_land_mask,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Whole-domain OAT diagnostic: info mode when only checkpoint given."
    )
    p.add_argument("checkpoint", help="Path to .ckpt file")
    p.add_argument(
        "--time",
        default=None,
        metavar="YYYY-MM-DD[:HH]",
        help="Snapshot time. HH defaults to 00 if omitted. "
        "Default: first validation timestep.",
    )
    p.add_argument(
        "--var",
        default=None,
        help="Variable to plot. Default: first variable in predictor.",
    )
    p.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Spatial grid resolution in coordinate units. "
        "Default: domain_width / 100 (targets ~100 grid points per axis).",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=50,
        metavar="N",
        help="Samples from the predictive distribution (default: 50).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output path. Default: data/figures/<tag>_domain_<var>_<time>.png",
    )
    p.add_argument(
        "--split",
        choices=["val", "test"],
        default=None,
        help=(
            "Station split mode. 'val': train stations as circles, "
            "val stations as squares. 'test': train+val as circles, "
            "test stations as squares."
        ),
    )
    p.add_argument(
        "--margin",
        type=float,
        default=0.05,
        metavar="F",
        help="Fractional margin around station bounding box (default: 0.05 = 5%%).",
    )
    p.add_argument(
        "--no-stations",
        action="store_true",
        default=False,
        help="Suppress all station markers from the plot.",
    )
    return p.parse_args()


def _resolve_time_idx(dataset, time_str: str) -> int:
    ts = pd.Timestamp(time_str.replace(":", " "), tz="UTC")
    idx = int(dataset.times.searchsorted(ts))
    return min(idx, len(dataset) - 1)


def _print_info(ckpt_path: str, predictor, dataset) -> None:
    ext = dataset.nwp_extent
    x_min, y_min = ext["min"][0].item(), ext["min"][1].item()
    x_max, y_max = ext["max"][0].item(), ext["max"][1].item()
    vars_str = ", ".join(predictor.var_to_indx.keys())
    print(f"Checkpoint : {ckpt_path}")
    print(f"Variables  : {vars_str}")
    print(
        f"Time range : {dataset.times[0]}  →  {dataset.times[-1]}"
        f"  ({len(dataset)} timesteps)"
    )
    print(
        f"NWP extent : X {x_min:.4g}–{x_max:.4g}  Y {y_min:.4g}–{y_max:.4g}"
        f"  (shape {dataset.nwp_shape})"
    )
    print(f"Stations   : {dataset.station_coords.shape[0]}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

    args = _parse_args()

    info_mode = (
        all(
            v is None
            for v in [args.time, args.var, args.resolution, args.out, args.split]
        )
        and args.n_samples == 50
    )

    predictor, cfg = load_checkpoint(OATPredictor, args.checkpoint, map_location="cpu")
    cfg.dataset.data_points_for_stats = 0
    predictor = predictor.eval()

    dataset, _, _, val_loader, test_loader, *_ = setup_data_and_transform(cfg, logger)

    if info_mode:
        _print_info(args.checkpoint, predictor, dataset)
        sys.exit(0)

    if is_field_path(dataset):
        logger.warning(
            "This script's domain overview compares against an NWP "
            "baseline, which is not yet supported on the Field data path; "
            "skipping."
        )
        sys.exit(0)

    obs_station_indices = None
    eval_indices = None
    if args.split is not None:
        split_loader = val_loader if args.split == "val" else test_loader
        split_dataset = split_loader.dataset
        obs_station_indices = list(split_dataset.observed_indices)  # type: ignore[attr-defined]
        eval_indices = list(split_dataset.eval_indices)  # type: ignore[attr-defined]
        logger.info(
            "Split '%s': %d observed, %d eval stations",
            args.split,
            len(obs_station_indices),
            len(eval_indices),
        )

    var: str = args.var if args.var is not None else next(iter(predictor.var_to_indx))

    if args.time is not None:
        snap_idx = _resolve_time_idx(dataset, args.time)
    else:
        snap_idx = (
            val_loader.dataset.indices[0]
            if hasattr(val_loader.dataset, "indices")
            else 0
        )
    snap_time_iso = dataset.times[snap_idx].strftime("%Y-%m-%dT%H")

    xs, ys = build_domain_grid(dataset, margin=args.margin, resolution=args.resolution)
    logger.info(
        "Domain grid: %dx%d (%d query points)",
        len(xs),
        len(ys),
        len(xs) * len(ys),
    )
    logger.info("Plotting %s at %s", var, snap_time_iso)

    PlotterCls = select_plotter_cls(dataset)
    plotter = PlotterCls(
        predictor=predictor,
        dataset=dataset,
        obs_station_indices=obs_station_indices,
        eval_indices=eval_indices,
        n_samples=args.n_samples,
        device="cpu",
        plot_config=load_plot_config(dataset),
    )

    nwp_predictor = maybe_build_nwp_baseline(dataset, logger, required=True)

    ckpt_tag = Path(args.checkpoint).parent.name
    fig = plotter.plot_domain_overview(
        other=nwp_predictor,
        var=var,
        time_idx=snap_idx,
        xs=xs,
        ys=ys,
        title=f"OAT domain overview — {var} — {snap_time_iso} — {ckpt_tag}",
        show_stations=not args.no_stations,
    )

    # Land-only masking and fine borders
    try:
        from matplotlib.colors import ListedColormap

        land_mask = build_land_mask(xs, ys)
        ocean_overlay = np.where(land_mask, np.nan, 1.0)
        data_axes = fig.axes[:4]
        for ax in data_axes:
            ax.pcolormesh(
                xs,
                ys,
                ocean_overlay,
                cmap=ListedColormap(["#c8e6f5"]),
                vmin=0.5,
                vmax=1.5,
                shading="auto",
                zorder=2,
            )
        _add_borders(data_axes, xs, ys)
        for ax in data_axes:
            ax.set_xlim(xs[0], xs[-1])
            ax.set_ylim(ys[0], ys[-1])
    except Exception as exc:
        logger.warning("Geo overlays skipped: %s", exc)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(args.out)
        if args.out is not None
        else OUT_DIR / f"{ckpt_tag}_domain_{var}_{snap_time_iso}.png"
    )
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out_path)


if __name__ == "__main__":
    print("Running OAT domain diagnostic...")
    main()
