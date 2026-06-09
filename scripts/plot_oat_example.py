"""OATPlotter diagnostic script.

Usage
-----
Info mode (no plotting — prints time range, spatial extent, variables):
    python -m scripts.plot_oat_example <checkpoint>

Plot mode (all optional args have sensible defaults):
    python -m scripts.plot_oat_example <checkpoint> --time 2020-06-15:06 \\
        --center 2683141 1248175 --hours 24 --extent 50000 --resolution 1000 \\
        --var temperature
"""
import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from oat.data_module.setup import setup_data_and_transform
from oat.model_module.baseline import NWPInterpolatorConfig
from oat.plotting_module import OATPlotter
from oat.predictor_module.baseline_predictor import NWPInterpolatorPredictor
from oat.predictor_module.oat_predictor import OATPredictor
from scripts.checkpoint_utils import load_checkpoint

OUT_DIR = Path(__file__).parents[1] / "data/figures"

_BERN_LV95 = (2_683_141.0, 1_248_175.0)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OATPlotter diagnostic: info mode when only checkpoint is given."
    )
    p.add_argument("checkpoint", help="Path to .ckpt file")
    loc_group = p.add_mutually_exclusive_group()
    loc_group.add_argument(
        "--center",
        nargs=2,
        type=float,
        metavar=("E", "N"),
        default=None,
        help="Centre position in Swiss LV95 metres (easting northing). "
        f"Default: Bern {_BERN_LV95}",
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
        help="Half-extent of spatial plot in metres (default: 50000)",
    )
    p.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Spatial grid resolution in metres (default: 1000)",
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
    return p.parse_args()


def _resolve_time_idx(dataset, time_str: str) -> int:
    ts = pd.Timestamp(time_str.replace(":", " "), tz="UTC")
    idx = int(dataset.times.searchsorted(ts))
    return min(idx, len(dataset) - 1)


def _print_info(ckpt_path: str, predictor, dataset) -> None:
    ext = dataset.nwp_extent
    e_min, n_min = ext["min"][0].item(), ext["min"][1].item()
    e_max, n_max = ext["max"][0].item(), ext["max"][1].item()
    vars_str = ", ".join(predictor.var_to_indx.keys())
    print(f"Checkpoint : {ckpt_path}")
    print(f"Variables  : {vars_str}")
    print(
        f"Time range : {dataset.times[0]}  →  {dataset.times[-1]}"
        f"  ({len(dataset)} timesteps)"
    )
    print(f"NWP extent : E {e_min:.0f}–{e_max:.0f} m  N {n_min:.0f}–{n_max:.0f} m")
    n_stations = dataset.station_coords.shape[0]
    print(f"Stations   : {n_stations}  (use --station <idx> to select one)")
    names = (
        dataset.station_names
        if hasattr(dataset, "station_names")
        else [str(i) for i in range(n_stations)]
    )
    for i, (name, coords) in enumerate(zip(names, dataset.station_coords)):
        e, n = coords[0].item(), coords[1].item()
        print(f"  [{i:>4d}] {name:<12s}  E={e:.0f}  N={n:.0f}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

    args = _parse_args()

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
    )

    predictor, cfg = load_checkpoint(OATPredictor, args.checkpoint)
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
            "Station %d (%s): E=%.0f N=%.0f", args.station, station_label, *center
        )
    else:
        center = tuple(args.center) if args.center is not None else _BERN_LV95
    hours: int = args.hours if args.hours is not None else 24
    extent: float = args.extent if args.extent is not None else 50_000.0
    resolution: float = args.resolution if args.resolution is not None else 1_000.0

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
    )

    nwp_config = NWPInterpolatorConfig(field_extent=dataset.nwp_extent)
    print(f"Observable variables: {dataset.config.obs_vars}")

    baseline_pre_transform = dataset.get_baseline_pre_transform()
    nwp_predictor = NWPInterpolatorPredictor(
        config=nwp_config,
        output_var_names=dataset.config.obs_vars,
        input_transform=baseline_pre_transform,
    ).eval()
    print("Var_idx:", nwp_predictor.var_to_indx)
    print("OAT predictor var_idx:", predictor.var_to_indx)
    nwp_plotter = OATPlotter(
        predictor=nwp_predictor,
        dataset=dataset,
        obs_station_indices=obs_station_indices,
        eval_indices=eval_indices,
        n_samples=1,
        device="cpu",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(30, 5))
    ckpt_tag = Path(args.checkpoint).parent.name
    fig.suptitle(f"OAT — {var} — {ckpt_tag}")

    if args.station is not None:
        plotter.plot_timeseries_at_station(
            ax=axes[0],
            station_idx=args.station,
            var=var,
            time_indices=time_indices,
            predictors={"NWP": nwp_predictor},
        )
        axes[0].set_title(
            f"Timeseries — {station_label} (E={center[0]:.0f}, N={center[1]:.0f})"
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
        )
        plt.colorbar(mesh_mean, ax=axes[1], label=var)
        axes[1].set_title(f"Mean  ({snap_time_iso})")
        axes[1].set_xlabel("Easting (m)")
        axes[1].set_ylabel("Northing (m)")

        mesh_std = plotter.plot_spatial_snapshot_at_station(
            ax=axes[2],
            var=var,
            time_idx=snap_idx,
            station_idx=args.station,
            extent=extent,
            resolution=resolution,
            plot_type="std",
        )

    else:
        plotter.plot_timeseries(
            ax=axes[0],
            center=center,
            var=var,
            time_indices=time_indices,
            predictors={"NWP": nwp_predictor},
        )
        axes[0].set_title(f"Timeseries (E={center[0]:.0f}, N={center[1]:.0f})")
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
        )
        plt.colorbar(mesh_mean, ax=axes[1], label=var)
        axes[1].set_title(f"Mean  ({snap_time_iso})")
        axes[1].set_xlabel("Easting (m)")
        axes[1].set_ylabel("Northing (m)")

        mesh_std = plotter.plot_spatial_snapshot(
            ax=axes[2],
            var=var,
            time_idx=snap_idx,
            center=center,
            extent=extent,
            resolution=resolution,
            plot_type="std",
        )
    plt.colorbar(mesh_std, ax=axes[2], label=f"{var} std")
    axes[2].set_title("Predictive std")
    axes[2].set_xlabel("Easting (m)")

    if args.station is not None:
        mesh_nwp = nwp_plotter.plot_spatial_snapshot_at_station(
            ax=axes[3], var=var, time_idx=snap_idx, station_idx=args.station,
            extent=extent, resolution=resolution, plot_type="mean",
        )
    else:
        mesh_nwp = nwp_plotter.plot_spatial_snapshot(
            ax=axes[3], var=var, time_idx=snap_idx, center=center,
            extent=extent, resolution=resolution, plot_type="mean",
        )
    plt.colorbar(mesh_nwp, ax=axes[3], label=var)
    axes[3].set_title(f"NWP interp.  ({snap_time_iso})")
    axes[3].set_xlabel("Easting (m)")

    if args.station is not None:
        mesh_diff = plotter.plot_spatial_difference_at_station(
            nwp_predictor,
            ax=axes[4],
            var=var,
            time_idx=snap_idx,
            station_idx=args.station,
            extent=extent,
            resolution=resolution,
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
        )
    plt.colorbar(mesh_diff, ax=axes[4], label=f"Δ{var}")
    axes[4].set_title(f"OAT − NWP  ({snap_time_iso})")
    axes[4].set_xlabel("Easting (m)")

    fig.tight_layout()
    out_path = OUT_DIR / f"{ckpt_tag}_{var}_{snap_time_iso}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Saved → %s", out_path)


if __name__ == "__main__":
    print("Running OATPlotter diagnostic...")
    main()
