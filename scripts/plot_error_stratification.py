"""Plot model error stratified by season, elevation, nearest-obs distance, lon/lat.

Requires per-(station, timestep) long dumps produced by
``scripts.evaluate_test_set --long`` for each model.

Usage
-----
python -m scripts.plot_error_stratification \
    --dataset cosmo_peakweather \
    --model oat_temp_precip \
    --baselines ootg_temp_precip,gp_temp \
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
    ckpt = resolve_checkpoint(dataset, model)
    tag = Path(ckpt).parent.name
    return OUT_DIR / f"{tag}_long_{split}.parquet"


def load_model_frames(
    dataset: str, model: str, baselines: list[str], split: str
) -> pd.DataFrame:
    """Load + concatenate long frames for the OAT model and its baselines."""
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
        df["model"] = key
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
    model_order = [args.model, *baselines]
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
