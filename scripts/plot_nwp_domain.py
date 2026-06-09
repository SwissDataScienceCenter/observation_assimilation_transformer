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
import torch

from oat.data_module.setup import setup_data_and_transform
from oat.model_module.baseline import NWPInterpolatorConfig
from oat.plotting_module import OATPlotter
from oat.predictor_module.baseline_predictor import NWPInterpolatorPredictor
from oat.predictor_module.oat_predictor import OATPredictor
from scripts.checkpoint_utils import load_checkpoint

OUT_DIR = Path(__file__).parents[1] / "data/figures"


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

    predictor, cfg = load_checkpoint(OATPredictor, args.checkpoint)
    cfg.dataset.data_points_for_stats = 0
    predictor = predictor.eval()

    dataset, _, _, val_loader, test_loader, *_ = setup_data_and_transform(cfg, logger)

    if info_mode:
        _print_info(args.checkpoint, predictor, dataset)
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

    # Topo validity scan: find the region where all topo channels are finite.
    # DEM has no NaN; TPI/derivative channels (1+) have NaN at the southern topo
    # boundary (~rows 0-1600 = ~80 km). Scan row validity within the NWP x-column
    # range; x only needs the patch half-size inset from the topo edge.
    ext = dataset.nwp_extent
    topo_ext = dataset.topo_extent
    patch_margin = dataset.topo_grid.abs().max().item()
    H, W = dataset.topo_field.shape[1:]
    topo_dx = (topo_ext["max"][0].item() - topo_ext["min"][0].item()) / (W - 1)
    topo_dy = (topo_ext["max"][1].item() - topo_ext["min"][1].item()) / (H - 1)
    col0 = max(0, int((ext["min"][0].item() - topo_ext["min"][0].item()) / topo_dx))
    col1 = min(
        W - 1, int((ext["max"][0].item() - topo_ext["min"][0].item()) / topo_dx) + 1
    )
    row0 = max(0, int((ext["min"][1].item() - topo_ext["min"][1].item()) / topo_dy))
    row1 = min(
        H - 1, int((ext["max"][1].item() - topo_ext["min"][1].item()) / topo_dy) + 1
    )
    stride = 10
    sub = dataset.topo_field[
        1:, row0:row1:stride, col0:col1:stride
    ]  # (C-1, n_rows, n_cols)
    row_valid = torch.isfinite(sub).all(dim=0).all(dim=1)
    valid_r = row_valid.nonzero(as_tuple=True)[0]
    extra = stride * topo_dy + patch_margin
    if len(valid_r):
        topo_y_min = (
            topo_ext["min"][1].item()
            + (row0 + valid_r[0].item() * stride) * topo_dy
            + extra
        )
        topo_y_max = (
            topo_ext["min"][1].item()
            + (row0 + valid_r[-1].item() * stride) * topo_dy
            - extra
        )
    else:
        topo_y_min, topo_y_max = ext["min"][1].item(), ext["max"][1].item()
    topo_x_min = topo_ext["min"][0].item() + patch_margin
    topo_x_max = topo_ext["max"][0].item() - patch_margin

    # Build query grid around station bounding box + fractional margin,
    # clamped to the valid topo region computed above.
    st_coords = dataset.station_coords.numpy()  # (N, 2)
    st_x_min, st_y_min = float(st_coords[:, 0].min()), float(st_coords[:, 1].min())
    st_x_max, st_y_max = float(st_coords[:, 0].max()), float(st_coords[:, 1].max())
    margin_x = (st_x_max - st_x_min) * args.margin
    margin_y = (st_y_max - st_y_min) * args.margin
    x_min = max(st_x_min - margin_x, topo_x_min)
    x_max = min(st_x_max + margin_x, topo_x_max)
    y_min = max(st_y_min - margin_y, topo_y_min)
    y_max = min(st_y_max + margin_y, topo_y_max)

    domain_width = x_max - x_min
    resolution: float = (
        args.resolution if args.resolution is not None else domain_width / 100.0
    )
    xs = np.arange(x_min, x_max + 1e-9, resolution)
    ys = np.arange(y_min, y_max + 1e-9, resolution)

    logger.info(
        "Domain grid: %dx%d (%d query points) at resolution %.4g",
        len(xs),
        len(ys),
        len(xs) * len(ys),
        resolution,
    )
    logger.info("Plotting %s at %s", var, snap_time_iso)

    plotter = OATPlotter(
        predictor=predictor,
        dataset=dataset,
        obs_station_indices=obs_station_indices,
        eval_indices=eval_indices,
        n_samples=args.n_samples,
        device="cpu",
    )

    nwp_config = NWPInterpolatorConfig(field_extent=dataset.nwp_extent)
    baseline_pre_transform = dataset.get_baseline_pre_transform()
    nwp_predictor = NWPInterpolatorPredictor(
        config=nwp_config,
        output_var_names=dataset.config.obs_vars,
        input_transform=baseline_pre_transform,
    ).eval()

    ckpt_tag = Path(args.checkpoint).parent.name
    fig = plotter.plot_domain_overview(
        other=nwp_predictor,
        var=var,
        time_idx=snap_idx,
        xs=xs,
        ys=ys,
        title=f"OAT domain overview — {var} — {snap_time_iso} — {ckpt_tag}",
    )

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
