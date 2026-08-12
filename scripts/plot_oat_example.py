"""OATPlotter diagnostic script.

Usage
-----
Info mode (no plotting — prints time range, spatial extent, variables):
    python -m scripts.plot_oat_example <checkpoint>

Plot mode (all optional args have sensible defaults):
    python -m scripts.plot_oat_example <checkpoint> --time 2020-06-15:06 \\
        --center 2683141 1248175 --hours 24 --extent 50000 --resolution 1000 \\
        --var temperature  (LV95 example)

    python -m scripts.plot_oat_example <checkpoint> --time 2020-06-15:06 \\
        --center 8.5 47.4 --hours 24 --extent 0.5 --resolution 0.01 \\
        --var temperature  (lon/lat example)
"""
import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from typing import Any

from oat.data_module.legacy.cosmo_peakweather_dataset import CosmoPeakWeatherDataset
from oat.data_module.legacy.era5_meteostat_dataset import ERA5MeteostatDataset
from oat.data_module.setup import setup_data_and_transform
from oat.model_module.baseline import NWPInterpolatorConfig
from oat.plotting_module import OATPlotter
from oat.predictor_module.baseline_predictor import NWPInterpolatorPredictor
from oat.predictor_module.checkpoint_utils import load_checkpoint, resolve_checkpoint
from oat.plotting_module.geo_overlays import _add_borders

OUT_DIR = Path(__file__).parents[1] / "data/figures"
_PROJECT_ROOT = Path(__file__).parents[1]
_PLOT_CONFIG_PATHS: dict[type, Path] = {
    ERA5MeteostatDataset: _PROJECT_ROOT / "configs/dataset/era5_meteostat/plot_config.yaml",
    CosmoPeakWeatherDataset: _PROJECT_ROOT / "configs/dataset/cosmo_peakweather/plot_config.yaml",
}


def _load_plot_config(dataset: Any) -> Any:
    path = _PLOT_CONFIG_PATHS.get(type(dataset))
    if path and path.exists():
        return OmegaConf.load(path).obs_plot_config
    return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OATPlotter diagnostic: info mode when only checkpoint is given."
    )
    p.add_argument("checkpoint", nargs="?", default=None, help="Path to .ckpt file")
    p.add_argument("--dataset", default=None, help="Dataset key in configs/checkpoints.yaml")
    p.add_argument("--model", default=None, help="Model key in configs/checkpoints.yaml")
    loc_group = p.add_mutually_exclusive_group()
    loc_group.add_argument(
        "--center",
        nargs=2,
        type=float,
        metavar=("E", "N"),
        default=None,
        help="Centre position in the dataset coordinate frame (x y). "
        "Default: midpoint of the NWP domain extent.",
    )
    loc_group.add_argument(
        "--station",
        type=int,
        default=None,
        metavar="IDX",
        help="Station index (see info mode for eligible stations). "
        "Sets the plot centre to the station's coordinates.",
    )
    p.add_argument(
        "--time",
        default=None,
        metavar="YYYY-MM-DD[:HH]",
        help="Snapshot start time. HH defaults to 00 if omitted. "
        "Default: first validation timestep.",
    )
    p.add_argument(
        "--hours",
        type=int,
        default=None,
        help="Timeseries length in hours (default: 24)",
    )
    p.add_argument(
        "--extent",
        type=float,
        default=None,
        help="Half-extent of spatial plot in coordinate units (default: 50000 for LV95, 0.5 for lon/lat).",
    )
    p.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Spatial grid resolution in coordinate units (default: 1000 for LV95, 0.01 for lon/lat).",
    )
    p.add_argument(
        "--var",
        default=None,
        help="Variable to plot. Default: first variable in predictor.",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=200,
        metavar="N",
        help="Number of samples drawn from the predictive distribution (default: 200).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Path to output figure. Default: data/figures/<checkpoint_folder>_<var>_<time>.png",
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
        help="Suppress all station markers from spatial plots.",
    )
    return p.parse_args()


def _resolve_time_idx(dataset, time_str: str) -> int:
    ts = pd.Timestamp(time_str.replace(":", " "), tz="UTC")
    idx = int(dataset.times.searchsorted(ts))
    if idx >= len(dataset) or dataset.times[idx] != ts:
        raise ValueError(
            f"Timestamp {ts} not found in dataset. "
            f"Range: {dataset.times[0]} – {dataset.times[-1]}"
        )
    return idx


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
    print(f"NWP extent : X {x_min:.4g}–{x_max:.4g}  Y {y_min:.4g}–{y_max:.4g}")
    n_stations = dataset.station_coords.shape[0]
    print(f"Stations   : {n_stations}  (use --station <idx> to select one)")
    names = (
        dataset.station_names
        if hasattr(dataset, "station_names")
        else [str(i) for i in range(n_stations)]
    )
    for i, (name, coords) in enumerate(zip(names, dataset.station_coords)):
        x, y = coords[0].item(), coords[1].item()
        print(f"  [{i:>4d}] {name:<12s}  X={x:.4g}  Y={y:.4g}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

    args = _parse_args()

    if args.checkpoint is None:
        if args.dataset is None or args.model is None:
            logger.error("Provide either a checkpoint path or both --dataset and --model.")
            raise SystemExit(2)
        args.checkpoint = resolve_checkpoint(args.dataset, args.model)

    info_mode = all(
        v is None
        for v in [
            args.center,
            args.station,
            args.time,
            args.hours,
            args.extent,
            args.resolution,
            args.var,
            args.split,
        ]
    ) and not args.no_stations

    predictor, cfg = load_checkpoint(None, args.checkpoint, map_location="cpu")
    cfg.dataset.data_points_for_stats = 0
    predictor = predictor.eval()

    dataset, num_train, _, val_loader, test_loader, *_ = setup_data_and_transform(
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
            "Split '%s': %d observed stations, %d eval stations",
            args.split,
            len(obs_station_indices),
            len(eval_indices),
        )

    # Apply defaults for plotting path
    var: str = args.var if args.var is not None else next(iter(predictor.var_to_indx))
    if args.station is not None:
        n_stations = dataset.station_coords.shape[0]
        if not 0 <= args.station < n_stations:
            raise SystemExit(
                f"--station {args.station} out of range (0–{n_stations - 1})"
            )
        coords = dataset.station_coords[args.station]
        center: tuple[float, float] = (coords[0].item(), coords[1].item())
        station_label = (
            dataset.station_names[args.station]
            if hasattr(dataset, "station_names")
            else str(args.station)
        )
        logger.info(
            "Station %d (%s): X=%.4g Y=%.4g", args.station, station_label, *center
        )
    else:
        if args.center is not None:
            center: tuple[float, float] = (args.center[0], args.center[1])
        else:
            ext = dataset.nwp_extent
            center = (
                (ext["min"][0].item() + ext["max"][0].item()) / 2.0,
                (ext["min"][1].item() + ext["max"][1].item()) / 2.0,
            )
    hours: int = args.hours if args.hours is not None else 24
    _is_lv95 = abs(dataset.nwp_extent["max"][0].item()) > 1e5
    extent: float = args.extent if args.extent is not None else (50_000.0 if _is_lv95 else 0.5)
    resolution: float = args.resolution if args.resolution is not None else (1_000.0 if _is_lv95 else 0.01)

    if args.time is not None:
        snap_idx = _resolve_time_idx(dataset, args.time)
    else:
        snap_idx = (
            val_loader.dataset.indices[0]
            if hasattr(val_loader.dataset, "indices")
            else 0
        )

    time_indices = list(range(snap_idx, snap_idx + hours))
    snap_time_iso = dataset.times[snap_idx].strftime("%Y-%m-%dT%H")
    logger.info(
        "Plotting %s  |  centre (%.0f, %.0f)  |  %s + %d h",
        var,
        center[0],
        center[1],
        snap_time_iso,
        hours,
    )

    plotter = OATPlotter(
        predictor=predictor,
        dataset=dataset,
        obs_station_indices=obs_station_indices,
        eval_indices=eval_indices,
        n_samples=args.n_samples,
        device="cpu",
        plot_config=_load_plot_config(dataset),
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
        plot_config=_load_plot_config(dataset),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(30, 5))
    ckpt_tag = Path(args.checkpoint).parent.name
    fig.suptitle(f"{ckpt_tag} — {var}")

    if args.station is not None:
        plotter.plot_timeseries_at_station(
            ax=axes[0],
            station_idx=args.station,
            var=var,
            time_indices=time_indices,
            predictors={"NWP": nwp_predictor},
        )
        axes[0].set_title(
            f"Timeseries — {station_label} (X={center[0]:.4g}, Y={center[1]:.4g})"
        )
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel(var)
        axes[0].legend()
        axes[0].tick_params(axis="x", rotation=30)

        mesh_mean = plotter.plot_spatial_snapshot_at_station(
            ax=axes[1],
            var=var,
            time_idx=snap_idx,
            station_idx=args.station,
            extent=extent,
            resolution=resolution,
            plot_type="mean",
            show_stations=not args.no_stations,
        )
        plt.colorbar(mesh_mean, ax=axes[1], label=var)
        axes[1].set_title(f"Mean  ({snap_time_iso})")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Y")

        mesh_std = plotter.plot_spatial_snapshot_at_station(
            ax=axes[2],
            var=var,
            time_idx=snap_idx,
            station_idx=args.station,
            extent=extent,
            resolution=resolution,
            plot_type="std",
            show_stations=not args.no_stations,
        )

    else:
        plotter.plot_timeseries(
            ax=axes[0],
            center=center,
            var=var,
            time_indices=time_indices,
            predictors={"NWP": nwp_predictor},
        )
        axes[0].set_title(f"Timeseries (X={center[0]:.4g}, Y={center[1]:.4g})")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel(var)
        axes[0].legend()
        axes[0].tick_params(axis="x", rotation=30)

        mesh_mean = plotter.plot_spatial_snapshot(
            ax=axes[1],
            var=var,
            time_idx=snap_idx,
            center=center,
            extent=extent,
            resolution=resolution,
            plot_type="mean",
            show_stations=not args.no_stations,
        )
        plt.colorbar(mesh_mean, ax=axes[1], label=var)
        axes[1].set_title(f"Mean  ({snap_time_iso})")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Y")

        mesh_std = plotter.plot_spatial_snapshot(
            ax=axes[2],
            var=var,
            time_idx=snap_idx,
            center=center,
            extent=extent,
            resolution=resolution,
            plot_type="std",
            show_stations=not args.no_stations,
        )
    plt.colorbar(mesh_std, ax=axes[2], label=f"{var} std")
    axes[2].set_title("Predictive std")
    axes[2].set_xlabel("X")

    if args.station is not None:
        mesh_nwp = nwp_plotter.plot_spatial_snapshot_at_station(
            ax=axes[3], var=var, time_idx=snap_idx, station_idx=args.station,
            extent=extent, resolution=resolution, plot_type="mean",
            show_stations=not args.no_stations,
        )
    else:
        mesh_nwp = nwp_plotter.plot_spatial_snapshot(
            ax=axes[3], var=var, time_idx=snap_idx, center=center,
            extent=extent, resolution=resolution, plot_type="mean",
            show_stations=not args.no_stations,
        )
    plt.colorbar(mesh_nwp, ax=axes[3], label=var)
    axes[3].set_title(f"NWP interp.  ({snap_time_iso})")
    axes[3].set_xlabel("X")

    if args.station is not None:
        mesh_diff = plotter.plot_spatial_difference_at_station(
            nwp_predictor,
            ax=axes[4],
            var=var,
            time_idx=snap_idx,
            station_idx=args.station,
            extent=extent,
            resolution=resolution,
            show_stations=not args.no_stations,
        )
    else:
        mesh_diff = plotter.plot_spatial_difference(
            nwp_predictor,
            ax=axes[4],
            var=var,
            time_idx=snap_idx,
            center=center,
            extent=extent,
            resolution=resolution,
            show_stations=not args.no_stations,
        )
    plt.colorbar(mesh_diff, ax=axes[4], label=f"Δ{var}")
    axes[4].set_title(f"OAT − NWP  ({snap_time_iso})")
    axes[4].set_xlabel("X")

    fig.tight_layout()

    try:
        xs = np.arange(center[0] - extent, center[0] + extent + 1e-9, resolution)
        ys = np.arange(center[1] - extent, center[1] + extent + 1e-9, resolution)
        _add_borders(list(axes[1:]), xs, ys)
        for ax in axes[1:]:
            ax.set_xlim(xs[0], xs[-1])
            ax.set_ylim(ys[0], ys[-1])
    except Exception as exc:
        logger.warning("Geo overlays skipped: %s", exc)

    out_path = OUT_DIR / f"{ckpt_tag}_{var}_{snap_time_iso}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Saved → %s", out_path)


if __name__ == "__main__":
    print("Running OATPlotter diagnostic...")
    main()
