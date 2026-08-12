"""Compare OAT and OOTG predicted mean fields on example timesteps.

Usage
-----
    python -m scripts.plot_oat_vs_ootg
    python -m scripts.plot_oat_vs_ootg --var temperature --times 2020-06-15:06 2020-06-16:12
"""
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oat.data_module.setup import setup_data_and_transform
from oat.plotting_module import OATPlotter
from oat.plotting_module.geo_overlays import _add_borders
from oat.predictor_module.checkpoint_utils import load_checkpoint

OUT_DIR = Path(__file__).parents[1] / "data/figures"

_DEFAULT_OAT_CKPT = (
    "/scratch/mch/cdonner/observation_assimilation_transformer/checkpoints/"
    "oat_L/oat_L_v1_20260622_job4686983/best-model-epoch=44-val_loss=0.1879.ckpt"
)
_DEFAULT_OOTG_CKPT = (
    "/scratch/mch/cdonner/observation_assimilation_transformer/checkpoints/"
    "debugging/ootg-baseline_cosmo_peakweather_temperature_precipitation_"
    "20260701_job4761712/epoch=19-val_loss=0.0866.ckpt"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot OAT vs OOTG predicted mean fields on two example timesteps."
    )
    p.add_argument("--oat-checkpoint", default=_DEFAULT_OAT_CKPT)
    p.add_argument("--ootg-checkpoint", default=_DEFAULT_OOTG_CKPT)
    p.add_argument("--var", default="temperature")
    p.add_argument(
        "--times",
        nargs=2,
        default=None,
        metavar=("TIME1", "TIME2"),
        help="Two snapshot times 'YYYY-MM-DD[:HH]'. Default: first two "
        "validation timesteps.",
    )
    p.add_argument(
        "--center",
        nargs=2,
        type=float,
        metavar=("E", "N"),
        default=None,
        help="Grid centre in the dataset coordinate frame. Default: midpoint "
        "of the NWP domain extent.",
    )
    p.add_argument("--extent", type=float, default=None)
    p.add_argument("--resolution", type=float, default=None)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--out", default=None)
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

    args = _parse_args()

    predictor_oat, cfg = load_checkpoint(None, args.oat_checkpoint, map_location="cpu")
    predictor_ootg, _ = load_checkpoint(None, args.ootg_checkpoint, map_location="cpu")
    predictor_oat = predictor_oat.eval()
    predictor_ootg = predictor_ootg.eval()

    cfg.dataset.data_points_for_stats = 0
    dataset, _, _, val_loader, _test_loader, *_ = setup_data_and_transform(cfg, logger)

    if args.times is not None:
        time_indices = [_resolve_time_idx(dataset, t) for t in args.times]
    else:
        base_idx = (
            val_loader.dataset.indices[0]
            if hasattr(val_loader.dataset, "indices")
            else 0
        )
        time_indices = [base_idx, base_idx + 1]

    if args.center is not None:
        center: tuple[float, float] = (args.center[0], args.center[1])
    else:
        ext = dataset.nwp_extent
        center = (
            (ext["min"][0].item() + ext["max"][0].item()) / 2.0,
            (ext["min"][1].item() + ext["max"][1].item()) / 2.0,
        )
    _is_lv95 = abs(dataset.nwp_extent["max"][0].item()) > 1e5
    extent = args.extent if args.extent is not None else (50_000.0 if _is_lv95 else 0.5)
    resolution = (
        args.resolution if args.resolution is not None else (1_000.0 if _is_lv95 else 0.01)
    )

    plotter_oat = OATPlotter(
        predictor=predictor_oat,
        dataset=dataset,
        n_samples=args.n_samples,
        device="cpu",
    )
    plotter_ootg = OATPlotter(
        predictor=predictor_ootg,
        dataset=dataset,
        n_samples=args.n_samples,
        device="cpu",
    )

    var = args.var
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    xs = np.arange(center[0] - extent, center[0] + extent + 1e-9, resolution)
    ys = np.arange(center[1] - extent, center[1] + extent + 1e-9, resolution)

    for row, time_idx in enumerate(time_indices):
        snap_time_iso = dataset.times[time_idx].strftime("%Y-%m-%dT%H")
        logger.info("Plotting %s at %s", var, snap_time_iso)

        for col, (name, plotter) in enumerate(
            [("OAT", plotter_oat), ("OOTG", plotter_ootg)]
        ):
            ax = axes[row, col]
            mesh = plotter.plot_spatial_snapshot(
                ax=ax,
                var=var,
                time_idx=time_idx,
                center=center,
                extent=extent,
                resolution=resolution,
                plot_type="mean",
                show_stations=True,
            )
            plt.colorbar(mesh, ax=ax, label=var)
            ax.set_title(f"{name} — {snap_time_iso}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

            field = mesh.get_array()
            logger.info(
                "  %s field range: [%.3g, %.3g], nan_frac=%.3f",
                name,
                np.nanmin(field),
                np.nanmax(field),
                np.isnan(field).mean(),
            )

    fig.tight_layout()

    try:
        _add_borders(list(axes.ravel()), xs, ys)
        for ax in axes.ravel():
            ax.set_xlim(xs[0], xs[-1])
            ax.set_ylim(ys[0], ys[-1])
    except Exception as exc:
        logger.warning("Geo overlays skipped: %s", exc)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    times_tag = "_".join(
        dataset.times[t].strftime("%Y%m%dT%H") for t in time_indices
    )
    out_path = Path(args.out) if args.out else OUT_DIR / f"oat_vs_ootg_{var}_{times_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
