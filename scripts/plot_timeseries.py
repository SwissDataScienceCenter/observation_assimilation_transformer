"""Timeseries figure: NWP interp vs pre-OAT vs OAT at a single location.

One row per variable, time on the x axis.  Each row shows:
  * NWP interpolation — deterministic baseline, median line only
  * Pre-OAT  — model output before the assimilation block (median + band)
  * OAT      — full model (median + band)
  * station observations, when the location is a station

Uses one OAT checkpoint plus an NWP-interpolator baseline built automatically
from the dataset.

Note on cost: each line runs one forward pass per timestep. Two separate calls
draw the two stages, so a 24 h window costs about 72 passes. Keep ``--hours``
modest, or lower ``--n-samples``.

Usage
-----
Resolve the checkpoint from configs/checkpoints.yaml; list the eligible test
stations for a window (no figure produced):
    python -m scripts.plot_timeseries --dataset cosmo_peakweather \\
        --model oat_temp_precip --time 2023-11-14:00 --hours 24

Then plot one of them, or an arbitrary point:
    python -m scripts.plot_timeseries --dataset cosmo_peakweather \\
        --model oat_temp_precip --station 42 --time 2023-11-14:00 --hours 24
    python -m scripts.plot_timeseries --dataset cosmo_peakweather \\
        --model oat_temp_precip --center 2660000 1190000 --hours 48

Or give a checkpoint path directly:
    python -m scripts.plot_timeseries <checkpoint> --station 42 --hours 24
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import OmegaConf

from oat.data_module.setup import setup_data_and_transform
from oat.plotting_module import OATPlotter
from oat.predictor_module.checkpoint_utils import load_checkpoint, resolve_checkpoint
from scripts.plotter_utils import load_plot_config, maybe_build_nwp_baseline

_PROJECT_ROOT = Path(__file__).parents[1]
OUT_DIR = _PROJECT_ROOT / "data/figures"

# ---- Figure sizing & fonts (single source; mirror in other paper figures) ----
# A fixed width makes the output size predictable and lets every figure share one font
# size: font points are absolute, so identical width + identical rcParams + the same
# include-width in LaTeX give identical on-page fonts. Unlike the map figures, these
# panels are not aspect-locked and carry no colorbar column, so the height is simply a
# fixed number of inches per variable row plus fixed margins in inches, which track the
# font size rather than the figure size.
FIG_WIDTH_IN = 16.0  # target figure width (inches) — keep identical across figures
_ROW_HEIGHT_IN = 3.6  # inches per variable row (the plotting area itself)
_MARGIN_H_IN = 1.4  # inches for panel titles + top/bottom margins
plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "lines.markersize": 5,
        "lines.linewidth": 2.0,
    }
)

# Fixed, semantic line colours so every row reads the same way, independent of
# the per-variable field colormaps used by the spatial figures. Each colour is
# used both for its median line and, at alpha 0.3, for its band, so it has to
# stay legible when faded and stay clear of grey (the NWP baseline) and black
# (the observations). This pair is the PRGn diverging endpoints: dark enough to
# fade well, and colour-blind safe.
_NWP_COLOR = "#5D5B5D"
_PRE_OAT_COLOR = "#7b3294"  # purple
_OAT_COLOR = "#008837"  # green


def _load_units(dataset_name: str) -> dict[str, str]:
    """Map observed-variable name -> unit string from the dataset catalog."""
    path = _PROJECT_ROOT / f"configs/dataset/{dataset_name}/catalog.yaml"
    if not path.exists():
        return {}
    obs = OmegaConf.load(path).get("obs_vars", {}) or {}
    return {v: str(obs[v].get("unit", "")) for v in obs}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help="Path to .ckpt file. Omit it to resolve the checkpoint "
        "with --dataset and --model.",
    )
    p.add_argument(
        "--dataset", default=None, help="Dataset key for the checkpoint registry."
    )
    p.add_argument(
        "--model", default=None, help="Model key for the checkpoint registry."
    )
    loc = p.add_mutually_exclusive_group()
    loc.add_argument(
        "--station",
        type=int,
        default=None,
        metavar="IDX",
        help="Station index; observations are overlaid. Omit both this and "
        "--center to print the eligible test stations for the window and exit.",
    )
    loc.add_argument(
        "--center",
        nargs=2,
        type=float,
        default=None,
        metavar=("E", "N"),
        help="Arbitrary location in the dataset coordinate frame (no observations).",
    )
    p.add_argument(
        "--allow-assimilated-station",
        action="store_true",
        help="Permit --station to name a station in the assimilation context.",
    )
    p.add_argument("--time", default=None, metavar="YYYY-MM-DD[:HH]")
    p.add_argument("--hours", type=int, default=24)
    p.add_argument("--variables", nargs="+", default=None)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--quantile-lo", type=float, default=0.05)
    p.add_argument("--quantile-hi", type=float, default=0.95)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.checkpoint is None and (args.dataset is None or args.model is None):
        p.error("--dataset and --model are required when no checkpoint path is given.")
    if args.hours < 1:
        p.error("--hours must be at least 1.")
    return args


def _resolve_main_checkpoint(args: argparse.Namespace) -> str:
    """Return the main checkpoint path from the positional arg or the registry."""
    if args.checkpoint is not None:
        return args.checkpoint
    return resolve_checkpoint(args.dataset, args.model)


def _resolve_time_idx(dataset, time_str: str) -> int:
    ts = pd.Timestamp(time_str.replace(":", " "), tz="UTC")
    idx = int(dataset.times.searchsorted(ts))
    return min(idx, len(dataset) - 1)


def _print_station_list(
    dataset,
    eval_indices: list[int],
    time_indices: list[int],
    variables: list[str],
    var_indices: list[int],
) -> None:
    """List the held-out test stations, with observation counts in the window.

    Printed when no location is given: the index alone is unguessable, and a
    station with no observations for a plotted variable yields an empty row, so
    the counts are what make the choice an informed one.
    """
    n_st = dataset.station_coords.shape[0]
    names = (
        dataset.station_names
        if hasattr(dataset, "station_names")
        else [str(i) for i in range(n_st)]
    )
    # [n_times, n_eval_stations, n_vars] -> observations per station per variable
    counts = (
        dataset.obs_mask_tensor[time_indices][:, eval_indices][:, :, var_indices]
        .sum(dim=0)
        .int()
    )

    n_times = len(time_indices)
    header = "  ".join(f"{v[:8]:>8s}" for v in variables)
    print(
        f"Held-out test stations, observations out of {n_times} timesteps "
        f"({dataset.times[time_indices[0]]:%Y-%m-%d %H:%M} → "
        f"{dataset.times[time_indices[-1]]:%Y-%m-%d %H:%M} UTC):"
    )
    print(f"  {'idx':>5s}  {'name':<14s}  {header}")
    shown = 0
    for row, s in enumerate(eval_indices):
        per_var = counts[row]
        if int(per_var.sum()) == 0:
            continue
        cells = "  ".join(f"{int(c):>8d}" for c in per_var)
        print(f"  [{s:>4d}] {names[s]:<14s}  {cells}")
        shown += 1
    skipped = len(eval_indices) - shown
    if skipped:
        print(f"  ... {skipped} test station(s) omitted: no observations in the window")
    print("\nRe-run with --station <idx> (or --center E N for an arbitrary point).")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)
    args = _parse_args()

    checkpoint_path = _resolve_main_checkpoint(args)
    predictor_oat, cfg = load_checkpoint(None, checkpoint_path, map_location="cpu")
    predictor_oat = predictor_oat.eval()

    cfg.dataset.data_points_for_stats = 0
    dataset, _, _, _val_loader, test_loader, *_ = setup_data_and_transform(cfg, logger)

    # load_from_checkpoint restores d_geo_bias but not the dataset-derived
    # project_index_coords callable. Re-attach it from the fresh dataset:
    # without it, ERA5 checkpoints crash with a geo-width shape mismatch once
    # inference runs (see 2026-08-17-sphere-index-coords.md).
    if hasattr(predictor_oat, "set_index_projection"):
        predictor_oat.set_index_projection(dataset)

    test_ds = test_loader.dataset
    # Cast to plain int: EvalDataset.observed_indices/eval_indices are 1-D
    # tensors, and a 0-dim tensor hashes by identity, not value, so a plain
    # int would never match one in a set (silently breaking the assimilated/
    # held-out checks below). OATPlotter applies the same cast internally.
    obs_station_indices = [int(i) for i in test_ds.observed_indices]
    eval_indices = [int(i) for i in test_ds.eval_indices]
    logger.info(
        "Test split: %d observed, %d eval stations",
        len(obs_station_indices),
        len(eval_indices),
    )
    # Report the held-out period so --time can be chosen without guessing.
    if test_ds._time_indices is not None and len(test_ds._time_indices):
        first, last = int(test_ds._time_indices[0]), int(test_ds._time_indices[-1])
        logger.info(
            "Test period: %s - %s (%d timesteps, dataset indices %d-%d)",
            dataset.times[first].strftime("%Y-%m-%d %H:%M UTC"),
            dataset.times[last].strftime("%Y-%m-%d %H:%M UTC"),
            len(test_ds._time_indices),
            first,
            last,
        )
    else:
        logger.info(
            "Test split has no temporal hold-out: all %d timesteps available "
            "(%s → %s)",
            len(dataset),
            dataset.times[0].strftime("%Y-%m-%d %H:%M UTC"),
            dataset.times[-1].strftime("%Y-%m-%d %H:%M UTC"),
        )

    nwp_predictor = maybe_build_nwp_baseline(dataset, logger, required=True)

    if args.time is not None:
        start_idx = _resolve_time_idx(dataset, args.time)
    elif test_ds._time_indices is not None and len(test_ds._time_indices):
        # Default to the first held-out test timestep, not dataset index 0
        # (which falls in the train period).
        start_idx = int(test_ds._time_indices[0])
    else:
        start_idx = 0
    time_indices = list(range(start_idx, min(start_idx + args.hours, len(dataset))))
    if len(time_indices) < args.hours:
        logger.warning(
            "Window truncated to %d h by the end of the dataset", len(time_indices)
        )
    start_iso = dataset.times[time_indices[0]].strftime("%Y-%m-%dT%H")

    variables = args.variables or list(predictor_oat.var_to_indx.keys())
    unknown_vars = [v for v in variables if v not in predictor_oat.var_to_indx]
    if unknown_vars:
        valid = ", ".join(sorted(predictor_oat.var_to_indx.keys()))
        raise ValueError(f"Unknown --variables {unknown_vars}. Valid names: {valid}.")
    plot_config = load_plot_config(dataset)
    units = _load_units(cfg.dataset.dataset_type)

    # With no location given, list the eligible stations and stop rather than
    # guessing one — the caller cannot be expected to know indices by heart.
    station_idx = args.station
    if station_idx is None and args.center is None:
        _print_station_list(
            dataset,
            eval_indices,
            time_indices,
            variables,
            [predictor_oat.var_to_indx[v] for v in variables],
        )
        return

    if station_idx is not None:
        # A station in the observation context has its own observation fed to the
        # model at every timestep, so the OAT line reproduces it rather than
        # predicting it. Require an explicit opt-in for that figure.
        assimilated = station_idx in set(obs_station_indices)
        if assimilated and not args.allow_assimilated_station:
            raise ValueError(
                f"Station {station_idx} is in the assimilation context (a "
                "train/val station), so its observation is an input to the model. "
                "Run without --station to list the held-out test stations, or "
                "pass --allow-assimilated-station."
            )
        if assimilated:
            logger.warning(
                "Station %d is assimilated: the OAT line at this location "
                "reproduces an input, it is not a prediction.",
                station_idx,
            )
        coords = dataset.station_coords[station_idx]
        center = (coords[0].item(), coords[1].item())
        role = (
            "assimilated"
            if assimilated
            else "held out" if station_idx in set(eval_indices) else "not in any split"
        )
        loc_label = f"Station {station_idx} ({role})"
        loc_tag = f"station{station_idx}"
    else:
        center = tuple(args.center)
        loc_label = f"X={center[0]:.6g}, Y={center[1]:.6g}"
        loc_tag = f"x{center[0]:.0f}_y{center[1]:.0f}"
    logger.info(
        "Variables (rows): %s at %s from %s for %d h",
        variables,
        loc_label,
        start_iso,
        len(time_indices),
    )

    plotter = OATPlotter(
        predictor=predictor_oat,
        dataset=dataset,
        obs_station_indices=obs_station_indices,
        eval_indices=eval_indices,
        n_samples=args.n_samples,
        device="cpu",
        plot_config=plot_config,
    )

    n_rows = len(variables)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(FIG_WIDTH_IN, _MARGIN_H_IN + n_rows * _ROW_HEIGHT_IN),
        squeeze=False,
        sharex=True,
        layout="constrained",
    )

    for row, var in enumerate(variables):
        ax = axes[row, 0]
        stages = [
            ("pre_oat", "Pre-OAT", _PRE_OAT_COLOR),
            ("oat", "OAT", _OAT_COLOR),
        ]
        for i, (stage, label, color) in enumerate(stages):
            # The deterministic NWP baseline is a property of the location, not
            # of the stage, so draw it once alongside the first stage. The
            # observations likewise go on once, with the last call so they sit
            # on top of both bands.
            kwargs = dict(
                var=var,
                time_indices=time_indices,
                quantile_lo=args.quantile_lo,
                quantile_hi=args.quantile_hi,
                predictors={"NWP interpolation": nwp_predictor} if i == 0 else None,
                stage=stage,
                label=label,
                color=color,
            )
            if station_idx is not None:
                plotter.plot_timeseries_at_station(
                    ax=ax,
                    station_idx=station_idx,
                    show_observations=(i == len(stages) - 1),
                    **kwargs,
                )
            else:
                plotter.plot_timeseries(ax=ax, center=center, **kwargs)

        unit = units.get(var, "")
        ax.set_ylabel(f"{var.capitalize()} [{unit}]" if unit else var.capitalize())
        ax.grid(alpha=0.3)
        if row == n_rows - 1:
            ax.set_xlabel("Time (UTC)")
            ax.tick_params(axis="x", rotation=30)

    # The NWP line is drawn by plot_timeseries without an explicit colour, so
    # recolour it here rather than duplicating the predictors loop.
    for ax in axes.ravel():
        for line in ax.get_lines():
            if line.get_label() == "NWP interpolation":
                line.set_color(_NWP_COLOR)
                line.set_linestyle("--")

    # Rows share the same series, so one figure-level legend suffices.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=min(3, len(handles)),
        frameon=False,
    )
    fig.suptitle(f"{loc_label} — {start_iso}, {len(time_indices)} h")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds_tag = args.dataset if args.dataset is not None else cfg.dataset.dataset_type
    model_tag = (
        args.model if args.model is not None else Path(checkpoint_path).parent.name
    )
    out_path = (
        Path(args.out)
        if args.out is not None
        else OUT_DIR / f"timeseries_{ds_tag}_{model_tag}_{loc_tag}_{start_iso}_"
        f"{len(time_indices)}h.png"
    )
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
