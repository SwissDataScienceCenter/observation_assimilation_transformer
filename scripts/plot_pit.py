"""PIT (Probability Integral Transform) calibration histogram.

For each evaluation station and timestep, computes the empirical CDF of the
predicted distribution at the observed value.  A calibrated model produces
PIT values that are uniform on [0, 1] — visible as a flat histogram.

Usage
-----
    python -m scripts.plot_pit <checkpoint> --split test --n-samples 1000 --n-timesteps 500 --no-title
"""
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from oat.data_module.setup import setup_data_and_transform
from oat.plotting_module import OATPlotter
from oat.predictor_module.checkpoint_utils import load_checkpoint
from scripts.plotter_utils import load_plot_config

OUT_DIR = Path(__file__).parents[1] / "data/figures"

MODEL_VAR_TO_FULL_VAR_NAME = {
    "temperature": "Temperature",
    "precipitation": "Precipitation",
}


def _parse_args():
    p = argparse.ArgumentParser(description="PIT calibration histogram per variable.")
    p.add_argument("checkpoint", help="Path to .ckpt file")
    p.add_argument("--split", choices=["val", "test"], default="test",
                   help="Which split to evaluate (default: test)")
    p.add_argument("--n-samples", type=int, default=200, metavar="N",
                   help="Samples drawn per query point (default: 200)")
    p.add_argument("--n-timesteps", type=int, default=None, metavar="N",
                   help="Subsample this many timesteps for speed (default: all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-title", action="store_true", default=False,
                   help="Suppress the figure title (for paper figures). The metadata "
                        "(split, timesteps, samples) is encoded in the output filename instead.")
    p.add_argument("--out", default=None, metavar="PATH",
                   help="Output path stem (extension ignored; .png and .pdf are both written). "
                        "Default: data/figures/<tag>_pit_<split>_<T>steps_<S>samples")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    args = _parse_args()

    predictor, cfg = load_checkpoint(None, args.checkpoint, map_location="cpu")
    cfg.dataset.data_points_for_stats = 0
    predictor = predictor.eval()

    dataset, _, _, val_loader, test_loader, *_ = setup_data_and_transform(cfg, logger)

    if hasattr(predictor, "set_index_projection"):
        predictor.set_index_projection(dataset)

    split_loader = val_loader if args.split == "val" else test_loader
    split_dataset = split_loader.dataset
    obs_station_indices = list(split_dataset.observed_indices)  # type: ignore[attr-defined]
    eval_indices = list(split_dataset.eval_indices)  # type: ignore[attr-defined]

    if not eval_indices:
        logger.error("No eval stations for split '%s' — cannot compute PIT.", args.split)
        return

    logger.info(
        "Split '%s': %d obs stations, %d eval stations",
        args.split, len(obs_station_indices), len(eval_indices),
    )

    # Evaluate ONLY the split's held-out time window, as positions into the BASE dataset
    # (which is what compute_pit indexes: self.dataset[time_idx], obs_tensor[time_idx]).
    # EvalDataset stores this window in `_time_indices`; with test_days>0 it is exactly the
    # test window. We deliberately do NOT fall back to range(len): that would evaluate the
    # first N base timesteps (inside the training period) instead of the test window.
    split_time_indices = getattr(split_dataset, "_time_indices", None)
    if split_time_indices is None:
        logger.error(
            "The '%s' split has no time window (_time_indices is None) — refusing to guess. "
            "This script requires a temporal split (test_days > 0).", args.split,
        )
        return
    time_indices: list[int] = [int(t) for t in split_time_indices]
    n_total = len(time_indices)

    if args.n_timesteps is not None and args.n_timesteps < n_total:
        rng = np.random.default_rng(args.seed)
        time_indices = sorted(
            int(t) for t in rng.choice(time_indices, size=args.n_timesteps, replace=False)
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

    all_data = plotter.compute_pit(time_indices)

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
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3),
                             gridspec_kw={"width_ratios": width_ratios, "wspace": 0.35})
    axes = np.atleast_1d(axes)

    ckpt_tag = Path(args.checkpoint).parent.name
    if not args.no_title:
        fig.suptitle(
            f"PIT calibration — {ckpt_tag}\n"
            f"{args.split} split  |  {len(time_indices)} timesteps  |  {args.n_samples} samples",
            y=1.03,
        )

    col = 0
    for var in vars_list:
        data = all_data[var]
        if var in zi_vars:
            ax_rel = axes[col]
            ax_pit = axes[col + 1]
            plotter.plot_zero_reliability(ax_rel, data["gate_pred"], data["is_zero"])
            ax_rel.set_title(f"{MODEL_VAR_TO_FULL_VAR_NAME.get(var, var)}\nzero probability", fontsize=9)
            # Use pit_nonzero (conditional on non-zero obs); swap for data["pit"] for full PIT.
            n_nz = len(data["pit_nonzero"])
            plotter.plot_pit(ax_pit, data["pit_nonzero"])
            ax_pit.set_title(f"{MODEL_VAR_TO_FULL_VAR_NAME.get(var, var)}\nPIT | non-zero  (n={n_nz:,})", fontsize=9)
            logger.info("%s: %d non-zero PIT values", var, n_nz)
            col += 2
        else:
            ax = axes[col]
            plotter.plot_pit(ax, data["pit"])
            ax.set_title(f"{MODEL_VAR_TO_FULL_VAR_NAME.get(var, var)}\n{ax.get_title()}", fontsize=9)
            logger.info("%s: %d PIT values", var, len(data["pit"]))
            col += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Metadata (split, #timesteps used, #samples) goes in the filename so it's not lost
    # when the on-figure title is suppressed for the paper.
    if args.out is not None:
        stem = Path(args.out).with_suffix("")
    else:
        stem = OUT_DIR / (
            f"{ckpt_tag}_pit_{args.split}_{len(time_indices)}steps_{args.n_samples}samples"
        )
    for ext in ("png", "pdf"):
        out_path = f"{stem}.{ext}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Saved → %s", out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
