"""PIT (Probability Integral Transform) calibration histogram.

For each evaluation station and timestep, computes the empirical CDF of the
predicted distribution at the observed value.  A calibrated model produces
PIT values that are uniform on [0, 1] — visible as a flat histogram.

Usage
-----
    # Resolve the checkpoint from configs/checkpoints.yaml
    python -m scripts.plot_pit --dataset cosmo_peakweather --model oat_temp_precip

    # Or give a checkpoint path directly, with a baseline outline on top
    python -m scripts.plot_pit <checkpoint> --baseline uninformed_temp_precip \\
        --dataset cosmo_peakweather --split test --n-samples 1000 --n-timesteps 500
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from oat.data_module.setup import setup_data_and_transform
from oat.plotting_module import OATPlotter
from oat.predictor_module.checkpoint_utils import (
    load_checkpoint,
    resolve_checkpoint,
)
from scripts.plotter_utils import load_plot_config

OUT_DIR = Path(__file__).parents[1] / "data/figures"

MODEL_VAR_TO_FULL_VAR_NAME = {
    "temperature": "Temperature",
    "precipitation": "Precipitation",
    "temp": "Temperature",
    "prcp": "Precipitation",
}

MODEL_LABEL = "OAT"
BASELINE_LABELS = {
    "uninformed_temp_precip": "Uninformed",
}
BASELINE_COLOR = "#C0392B"

N_BINS = 20


def _parse_args():
    p = argparse.ArgumentParser(description="PIT calibration histogram per variable.")
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
    p.add_argument(
        "--baseline",
        default=None,
        help="Baseline model: a registry key (resolved with --dataset) "
        "or a .ckpt path. Its PIT is drawn as an outline over the bars.",
    )
    p.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Which split to evaluate (default: test)",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=200,
        metavar="N",
        help="Samples drawn per query point (default: 200)",
    )
    p.add_argument(
        "--n-timesteps",
        type=int,
        default=None,
        metavar="N",
        help="Subsample this many timesteps for speed (default: all)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Output path stem (extension ignored; .png and .pdf are both written). "
        "Default: data/figures/<tag>_pit_<split>_<T>steps_<S>samples",
    )
    args = p.parse_args()
    if args.checkpoint is None and (args.dataset is None or args.model is None):
        p.error("--dataset and --model are required when no checkpoint path is given.")
    return args


def _resolve_main_checkpoint(args: argparse.Namespace) -> str:
    """Return the main checkpoint path from the positional arg or the registry."""
    if args.checkpoint is not None:
        return args.checkpoint
    return resolve_checkpoint(args.dataset, args.model)


def _resolve_baseline_checkpoint(baseline: str, dataset: str | None) -> tuple[str, str]:
    """Return (path, label) for the baseline from a registry key or a file path."""
    if Path(baseline).exists():
        return baseline, Path(baseline).parent.name
    if dataset is None:
        raise ValueError(
            f"Baseline {baseline!r} is not a file. Give --dataset to resolve it "
            "as a registry key."
        )
    try:
        return resolve_checkpoint(dataset, baseline), baseline
    except KeyError:
        raise ValueError(
            f"Baseline {baseline!r} is not a file and not a registry key for "
            f"dataset={dataset!r}."
        ) from None


def _plot_baseline_outline(
    plotter: OATPlotter,
    ax,
    baseline_data: dict | None,
    var: str,
    conditional: bool,
    label: str,
) -> None:
    """Draw the baseline PIT as an outline on ax. Draw nothing if it has no data."""
    if baseline_data is None or var not in baseline_data:
        return
    b = baseline_data[var]
    if conditional:
        counts = b.get("pit_nonzero_counts")
        n = b.get("pit_nonzero_n", 0)
    else:
        counts = b["pit_counts"]
        n = b["pit_n"]
    if counts is None or n == 0:
        return
    plotter.plot_pit(ax, counts, n_bins=N_BINS, total=n, histtype="step", label=label)


def _plot_baseline_reliability(ax, baseline_data: dict | None, var: str, label: str):
    """Draw the baseline zero-gate reliability curve on ax, if it has data."""
    if baseline_data is None or var not in baseline_data:
        return
    b = baseline_data[var]
    counts = b.get("gate_counts")
    if counts is None:
        return
    ok = counts > 5
    if not ok.any():
        return
    ax.plot(
        b["gate_pred_sums"][ok] / counts[ok],
        b["gate_zero_sums"][ok] / counts[ok],
        "o-",
        color=BASELINE_COLOR,
        ms=4,
        lw=1.2,
        label=label,
    )


def _set_legend(ax, order: list[tuple[str, str]], fontsize: int = 8) -> None:
    """Show one legend entry per (source label, shown label) pair, in order."""
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=True))
    kept = [(src, shown) for src, shown in order if src in by_label]
    ax.legend(
        [by_label[src] for src, _ in kept],
        [shown for _, shown in kept],
        fontsize=fontsize,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)
    args = _parse_args()

    checkpoint_path = _resolve_main_checkpoint(args)
    predictor, cfg = load_checkpoint(None, checkpoint_path, map_location="cpu")
    cfg.dataset.data_points_for_stats = 0
    predictor = predictor.eval()

    dataset, _, _, val_loader, test_loader, *_ = setup_data_and_transform(cfg, logger)

    if hasattr(predictor, "set_index_projection"):
        predictor.set_index_projection(dataset)

    baseline_predictor = None
    baseline_tag = None
    if args.baseline is not None:
        baseline_path, baseline_tag = _resolve_baseline_checkpoint(
            args.baseline, args.dataset
        )
        baseline_predictor, _ = load_checkpoint(None, baseline_path, map_location="cpu")
        baseline_predictor = baseline_predictor.eval()
        if hasattr(baseline_predictor, "set_index_projection"):
            baseline_predictor.set_index_projection(dataset)
        logger.info("Baseline: %s", baseline_path)

    split_loader = val_loader if args.split == "val" else test_loader
    split_dataset = split_loader.dataset
    obs_station_indices = list(split_dataset.observed_indices)  # type: ignore[attr-defined]
    eval_indices = list(split_dataset.eval_indices)  # type: ignore[attr-defined]

    if not eval_indices:
        logger.error(
            "No eval stations for split '%s' — cannot compute PIT.", args.split
        )
        return

    logger.info(
        "Split '%s': %d obs stations, %d eval stations",
        args.split,
        len(obs_station_indices),
        len(eval_indices),
    )

    # Evaluate ONLY the split's held-out time window, as positions into the BASE
    # dataset (which is what compute_pit indexes: self.dataset[time_idx],
    # obs_tensor[time_idx]). EvalDataset stores this window in `_time_indices`;
    # with test_days>0 it is exactly the test window. We deliberately do NOT
    # fall back to range(len): that would evaluate the first N base timesteps
    # (inside the training period) instead of the test window.
    split_time_indices = getattr(split_dataset, "_time_indices", None)
    if split_time_indices is None:
        logger.error(
            "The '%s' split has no time window (_time_indices is None) — "
            "refusing to guess. This script requires a temporal split "
            "(test_days > 0).",
            args.split,
        )
        return
    time_indices: list[int] = [int(t) for t in split_time_indices]
    n_total = len(time_indices)

    if args.n_timesteps is not None and args.n_timesteps < n_total:
        rng = np.random.default_rng(args.seed)
        time_indices = sorted(
            int(t)
            for t in rng.choice(time_indices, size=args.n_timesteps, replace=False)
        )
        logger.info("Subsampled to %d / %d timesteps", len(time_indices), n_total)

    plotter = OATPlotter(
        predictor=predictor,
        dataset=dataset,
        obs_station_indices=obs_station_indices,
        eval_indices=eval_indices,
        n_samples=args.n_samples,
        device="cpu",
        plot_config=load_plot_config(dataset),
    )

    main_data = plotter.compute_pit(time_indices, n_bins=N_BINS)
    baseline_data = None
    if baseline_predictor is not None:
        baseline_data = plotter.compute_pit(
            time_indices, n_bins=N_BINS, predictor=baseline_predictor
        )

    try:
        var_to_dist = predictor.model.output_head.variable_to_distribution
    except AttributeError:
        var_to_dist = {}
    zi_vars = {v for v, d in var_to_dist.items() if d == "zero_inflated_lognormal"}
    vars_list = list(predictor.var_to_indx.keys())

    # ZI variables get two columns (reliability diagram + conditional PIT);
    # regular variables get one.
    col_counts = [2 if v in zi_vars else 1 for v in vars_list]
    n_cols = sum(col_counts)
    width_ratios = [3.5 for v in vars_list for _ in range(2 if v in zi_vars else 1)]
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(3.5 * n_cols, 3),
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.35},
    )
    axes = np.atleast_1d(axes)

    ckpt_tag = Path(checkpoint_path).parent.name
    baseline_label = BASELINE_LABELS.get(baseline_tag or "", baseline_tag or "")
    pit_legend = [
        (MODEL_LABEL, MODEL_LABEL),
        (baseline_label, baseline_label),
        ("Uniform", "Uniform"),
    ]

    col = 0
    for var in vars_list:
        data = main_data[var]
        if var in zi_vars:
            ax_rel = axes[col]
            ax_pit = axes[col + 1]
            plotter.plot_zero_reliability(
                ax_rel,
                data["gate_counts"],
                data["gate_zero_sums"],
                data["gate_pred_sums"],
            )
            if baseline_tag is not None:
                _plot_baseline_reliability(ax_rel, baseline_data, var, baseline_label)
            _set_legend(
                ax_rel,
                [
                    ("Model", MODEL_LABEL),
                    (baseline_label, baseline_label),
                    ("Perfect", "Perfect"),
                ],
            )
            ax_rel.set_title(
                f"{MODEL_VAR_TO_FULL_VAR_NAME.get(var, var)}\nzero probability",
                fontsize=9,
            )
            # Use pit_nonzero (conditional on non-zero obs).
            n_nz = data["pit_nonzero_n"]
            plotter.plot_pit(
                ax_pit,
                data["pit_nonzero_counts"],
                n_bins=N_BINS,
                total=n_nz,
                label=MODEL_LABEL,
            )
            if baseline_tag is not None:
                _plot_baseline_outline(
                    plotter,
                    ax_pit,
                    baseline_data,
                    var,
                    conditional=True,
                    label=baseline_label,
                )
            _set_legend(ax_pit, pit_legend)
            ax_pit.set_title(
                f"{MODEL_VAR_TO_FULL_VAR_NAME.get(var, var)}\n"
                f"PIT | non-zero  (n={n_nz:,})",
                fontsize=9,
            )
            logger.info("%s: %d non-zero PIT values", var, n_nz)
            col += 2
        else:
            ax = axes[col]
            plotter.plot_pit(
                ax,
                data["pit_counts"],
                n_bins=N_BINS,
                total=data["pit_n"],
                label=MODEL_LABEL,
            )
            if baseline_tag is not None:
                _plot_baseline_outline(
                    plotter,
                    ax,
                    baseline_data,
                    var,
                    conditional=False,
                    label=baseline_label,
                )
            _set_legend(ax, pit_legend)
            ax.set_title(
                f"{MODEL_VAR_TO_FULL_VAR_NAME.get(var, var)}\n{ax.get_title()}",
                fontsize=9,
            )
            logger.info("%s: %d PIT values", var, data["pit_n"])
            col += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Metadata (split, #timesteps used, #samples) goes in the filename so it's not lost
    # when the on-figure title is suppressed for the paper.
    if args.out is not None:
        stem = Path(args.out).with_suffix("")
    else:
        name = (
            f"{ckpt_tag}_pit_{args.split}_{len(time_indices)}steps"
            f"_{args.n_samples}samples"
        )
        if baseline_tag is not None:
            name += f"_vs_{baseline_tag}"
        stem = OUT_DIR / name
    for ext in ("png", "pdf"):
        out_path = f"{stem}.{ext}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Saved → %s", out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
