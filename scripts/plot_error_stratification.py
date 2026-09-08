"""Plot model error stratified by season, elevation, nearest-obs distance.

Requires per-(station, timestep) long dumps produced by
``scripts.evaluate_test_set --long`` for each model.

Usage
-----
python -m scripts.plot_error_stratification \
    --dataset cosmo_peakweather \
    --model oat_temp_precip \
    --baselines ootg_temp_precip,gp_oat_temp \
    --metric crps --split test
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from oat.plotting_module.stratification import make_stratification_figure
from oat.predictor_module.checkpoint_utils import resolve_checkpoint

OUT_DIR = Path(__file__).parents[1] / "data/results"
FIG_DIR = Path(__file__).parents[1] / "data/figures"


def long_parquet_path(dataset: str, model: str, split: str) -> Path:
    """Map a registry key to the long-dump Parquet path written by --long."""
    if model == "interpolator":
        tag = f"interpolator_{dataset}"
    else:
        ckpt = resolve_checkpoint(dataset, model)
        tag = Path(ckpt).parent.name
    return OUT_DIR / f"{tag}_long_{split}.parquet"


def group_name(key: str) -> str:
    """Collapse a per-variable registry key onto its model name.

    Per-variable checkpoints (e.g. ``gp_oat_temp``/``gp_oat_precip``,
    ``deep_gp_oat_temp``/``deep_gp_oat_precip``) are trained one variable each; they
    are treated as a single model in the plot. The combined ``*_temp_precip``
    keys map to the same bare model name. Each frame only carries finite values
    for its own variable, so concatenating them under one model name yields the
    full-variable model without further merging.
    """
    for suffix in ("_temp_precip", "_temp", "_precip"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def load_model_frames(
    dataset: str, model: str, baselines: list[str], split: str
) -> pd.DataFrame:
    """Load + concatenate long frames for the OAT model and its baselines.

    Keys sharing a model name (see ``group_name``) are relabelled to that name
    so per-variable checkpoints appear as a single model.
    """
    frames = []
    for key in [model, *baselines]:
        path = long_parquet_path(dataset, key, split)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing long dump for model={key!r}: {path}\n"
                f"Run: python -m scripts.evaluate_test_set --dataset {dataset} "
                f"--model {key} --split {split} --long"
            )
        df = pd.read_parquet(path)
        df["model"] = group_name(key)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _var_names(df: pd.DataFrame, metric: str) -> list[str]:
    prefix = f"{metric}_"
    return [c[len(prefix) :] for c in df.columns if c.startswith(prefix)]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True, help="Registry key for the OAT model")
    p.add_argument(
        "--baselines",
        required=True,
        help="Two comma-separated registry keys",
    )
    p.add_argument("--metric", choices=["crps", "mae"], default="crps")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--nbins", type=int, default=5)
    p.add_argument("--out", default=None, help="Output path stem (no extension)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)
    args = _parse_args()

    baselines = [b for b in args.baselines.split(",") if b]
    df = load_model_frames(args.dataset, args.model, baselines, args.split)
    # Dedup group names while preserving first-seen order (per-variable keys
    # collapse onto a single model, e.g. gp_oat_temp + gp_oat_precip -> gp_oat).
    model_order = list(dict.fromkeys(group_name(k) for k in [args.model, *baselines]))
    var_names = _var_names(df, args.metric)

    fig = make_stratification_figure(
        df,
        var_names=var_names,
        metric=args.metric,
        model_order=model_order,
        nbins=args.nbins,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = args.out or str(
        FIG_DIR / f"error_stratification_{args.dataset}_{args.metric}_{args.split}"
    )
    for ext in ("png", "pdf"):
        path = f"{stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved → %s", path)


if __name__ == "__main__":
    main()
