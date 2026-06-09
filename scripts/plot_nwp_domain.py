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


def _overlay_stations(
    ax,
    dataset,
    time_idx: int,
    var_idx: int,
    obs_indices: list[int],
    eval_indices: list[int],
) -> None:
    obs_set = set(obs_indices)
    eval_set = set(eval_indices)
    n_st = dataset.station_coords.shape[0]
    obs_x, obs_y, obs_vals = [], [], []
    missing_obs_x, missing_obs_y = [], []
    sq_x, sq_y = [], []

    for s in range(n_st):
        sx = dataset.station_coords[s, 0].item()
        sy = dataset.station_coords[s, 1].item()
        mask_val = dataset.obs_mask_tensor[time_idx, s, var_idx].item()
        if s in obs_set:
            if mask_val:
                obs_x.append(sx)
                obs_y.append(sy)
                obs_vals.append(dataset.obs_tensor[time_idx, s, var_idx].item())
            elif eval_set:
                missing_obs_x.append(sx)
                missing_obs_y.append(sy)
            else:
                sq_x.append(sx)
                sq_y.append(sy)
        elif s in eval_set:
            sq_x.append(sx)
            sq_y.append(sy)
        elif not eval_set:
            sq_x.append(sx)
            sq_y.append(sy)

    if obs_x:
        ax.scatter(obs_x, obs_y, marker="o", c="white", edgecolors="black",
                   s=10, zorder=5, label="observed")
    if missing_obs_x:
        ax.scatter(missing_obs_x, missing_obs_y, marker="o", facecolors="white",
                   edgecolors="grey", s=10, zorder=5, label="missing")
    sq_label = "eval" if eval_set else "not observed"
    if sq_x:
        ax.scatter(sq_x, sq_y, marker="s", facecolors="white", edgecolors="black",
                   s=10, zorder=5, label=sq_label)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger(__name__)

    args = _parse_args()

    info_mode = all(
        v is None for v in [args.time, args.var, args.resolution, args.out, args.split]
    ) and args.n_samples == 50

    predictor, cfg = load_checkpoint(OATPredictor, args.checkpoint)
    cfg.dataset.data_points_for_stats = 0
    predictor = predictor.eval()

    dataset, _, _, val_loader, test_loader, *_ = setup_data_and_transform(
        cfg, logger
    )

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

    # Build domain grid restricted to where all topo channels are valid.
    # DEM has no NaN; TPI/derivative channels (1+) have NaN at the southern topo
    # boundary (~rows 0-1600 = ~80 km). Clip y by scanning row validity within the
    # NWP x-column range. x only needs the patch half-size inset from the topo edge.
    ext = dataset.nwp_extent
    topo_ext = dataset.topo_extent
    patch_margin = dataset.topo_grid.abs().max().item()
    H, W = dataset.topo_field.shape[1:]
    topo_dx = (topo_ext["max"][0].item() - topo_ext["min"][0].item()) / (W - 1)
    topo_dy = (topo_ext["max"][1].item() - topo_ext["min"][1].item()) / (H - 1)
    # Map NWP extent to topo row/col indices
    col0 = max(0, int((ext["min"][0].item() - topo_ext["min"][0].item()) / topo_dx))
    col1 = min(W - 1, int((ext["max"][0].item() - topo_ext["min"][0].item()) / topo_dx) + 1)
    row0 = max(0, int((ext["min"][1].item() - topo_ext["min"][1].item()) / topo_dy))
    row1 = min(H - 1, int((ext["max"][1].item() - topo_ext["min"][1].item()) / topo_dy) + 1)
    # Scan non-DEM channels at stride 10 for speed
    stride = 10
    sub = dataset.topo_field[1:, row0:row1:stride, col0:col1:stride]  # (C-1, n_rows, n_cols)
    row_valid = torch.isfinite(sub).all(dim=0).all(dim=1)  # True where ALL cols are finite
    valid_r = row_valid.nonzero(as_tuple=True)[0]
    # Extra margin: stride uncertainty + patch half-size
    extra = stride * topo_dy + patch_margin
    if len(valid_r):
        y_min = max(ext["min"][1].item(),
                    topo_ext["min"][1].item() + (row0 + valid_r[0].item() * stride) * topo_dy + extra)
        y_max = min(ext["max"][1].item(),
                    topo_ext["min"][1].item() + (row0 + valid_r[-1].item() * stride) * topo_dy - extra)
    else:
        y_min, y_max = ext["min"][1].item(), ext["max"][1].item()
    # x: inset by patch half-size from topo boundary only
    x_min = max(ext["min"][0].item(), topo_ext["min"][0].item() + patch_margin)
    x_max = min(ext["max"][0].item(), topo_ext["max"][0].item() - patch_margin)
    domain_width = x_max - x_min
    resolution: float = args.resolution if args.resolution is not None else domain_width / 100.0
    xs = np.arange(x_min, x_max + 1e-9, resolution)
    ys = np.arange(y_min, y_max + 1e-9, resolution)
    x_grid, y_grid = np.meshgrid(xs, ys)
    query_coords = torch.tensor(
        np.stack([x_grid.ravel(), y_grid.ravel()], axis=1), dtype=torch.float32
    )

    logger.info(
        "Domain grid: %dx%d (%d query points) at resolution %.4g",
        len(xs), len(ys), len(xs) * len(ys), resolution,
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
    nwp_plotter = OATPlotter(
        predictor=nwp_predictor,
        dataset=dataset,
        obs_station_indices=obs_station_indices,
        eval_indices=eval_indices,
        n_samples=1,
        device="cpu",
    )

    logger.info("Running NWP inference...")
    samples_nwp = nwp_plotter._run_inference(snap_idx, query_coords, predictor=nwp_predictor)
    nwp_mean = samples_nwp[:, :, var_idx].mean(dim=1).numpy().reshape(y_grid.shape)

    logger.info("Running OAT inference...")
    samples_oat = plotter._run_inference(snap_idx, query_coords)
    oat_mean = samples_oat[:, :, var_idx].mean(dim=1).numpy().reshape(y_grid.shape)
    oat_std = samples_oat[:, :, var_idx].std(dim=1).numpy().reshape(y_grid.shape)

    diff = oat_mean - nwp_mean
    vmax_diff = float(np.abs(diff).max()) or 1.0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_tag = Path(args.checkpoint).parent.name
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle(f"OAT domain overview — {var} — {snap_time_iso} — {ckpt_tag}")

    obs_indices_list = plotter.obs_indices
    eval_indices_list = plotter.eval_indices

    # Panel 1: NWP interpolation
    mesh_nwp = axes[0].pcolormesh(xs, ys, nwp_mean, shading="auto")
    plt.colorbar(mesh_nwp, ax=axes[0], label=var)
    axes[0].set_title(f"NWP interp.  ({snap_time_iso})")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    _overlay_stations(axes[0], dataset, snap_idx, var_idx, obs_indices_list, eval_indices_list)

    # Panel 2: OAT mean
    mesh_mean = axes[1].pcolormesh(xs, ys, oat_mean, shading="auto",
                                    vmin=mesh_nwp.norm.vmin, vmax=mesh_nwp.norm.vmax,
                                    cmap=mesh_nwp.cmap)
    plt.colorbar(mesh_mean, ax=axes[1], label=var)
    axes[1].set_title("OAT mean")
    axes[1].set_xlabel("X")
    _overlay_stations(axes[1], dataset, snap_idx, var_idx, obs_indices_list, eval_indices_list)

    # Panel 3: OAT std
    mesh_std = axes[2].pcolormesh(xs, ys, oat_std, shading="auto")
    plt.colorbar(mesh_std, ax=axes[2], label=f"{var} std")
    axes[2].set_title("OAT predictive std")
    axes[2].set_xlabel("X")
    _overlay_stations(axes[2], dataset, snap_idx, var_idx, obs_indices_list, eval_indices_list)

    # Panel 4: OAT − NWP diff
    mesh_diff = axes[3].pcolormesh(xs, ys, diff, cmap="RdBu_r",
                                    vmin=-vmax_diff, vmax=vmax_diff, shading="auto")
    plt.colorbar(mesh_diff, ax=axes[3], label=f"Δ{var}")
    axes[3].set_title("OAT − NWP")
    axes[3].set_xlabel("X")
    _overlay_stations(axes[3], dataset, snap_idx, var_idx, obs_indices_list, eval_indices_list)

    fig.tight_layout()
    out_path = (
        Path(args.out)
        if args.out is not None
        else OUT_DIR / f"{ckpt_tag}_domain_{var}_{snap_time_iso}.png"
    )
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    logger.info("Saved → %s", out_path)


if __name__ == "__main__":
    print("Running OAT domain diagnostic...")
    main()
