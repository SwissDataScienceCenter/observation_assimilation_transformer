"""PIT (Probability Integral Transform) calibration histogram.

For each evaluation station and timestep, computes the empirical CDF of the
predicted distribution at the observed value.  A calibrated model produces
PIT values that are uniform on [0, 1] — visible as a flat histogram.

Usage
-----
    python -m scripts.plot_pit <checkpoint>
    python -m scripts.plot_pit <checkpoint> --split val --n-timesteps 500
"""
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from oat.data_module.cosmo_peakweather_dataset import CosmoPeakWeatherDataset
from oat.data_module.era5_meteostat_dataset import ERA5MeteostatDataset
from oat.data_module.setup import setup_data_and_transform
from oat.plotting_module import OATPlotter
from oat.predictor_module.checkpoint_utils import load_checkpoint
from oat.predictor_module.oat_predictor import OATPredictor

OUT_DIR = Path(__file__).parents[1] / "data/figures"
_PROJECT_ROOT = Path(__file__).parents[1]
_PLOT_CONFIG_PATHS = {
    ERA5MeteostatDataset: _PROJECT_ROOT / "configs/dataset/era5_meteostat/plot_config.yaml",
    CosmoPeakWeatherDataset: _PROJECT_ROOT / "configs/dataset/cosmo_peakweather/plot_config.yaml",
}


def _load_plot_config(dataset):
    path = _PLOT_CONFIG_PATHS.get(type(dataset))
    if path and path.exists():
        return OmegaConf.load(path).obs_plot_config
    return None


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
    p.add_argument("--out", default=None, metavar="PATH",
                   help="Output figure path (default: data/figures/<tag>_pit_<split>.png)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    args = _parse_args()

    predictor, cfg = load_checkpoint(OATPredictor, args.checkpoint, map_location="cpu")
    cfg.dataset.data_points_for_stats = 0
    predictor = predictor.eval()

    dataset, _, _, val_loader, test_loader, *_ = setup_data_and_transform(cfg, logger)

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

    # Time indices for this split
    time_indices: list[int] = (
        list(split_dataset.indices)
        if hasattr(split_dataset, "indices")
        else list(range(len(split_dataset)))
    )

    if args.n_timesteps is not None and args.n_timesteps < len(time_indices):
        rng = np.random.default_rng(args.seed)
        time_indices = sorted(
            rng.choice(time_indices, size=args.n_timesteps, replace=False).tolist()
        )
        logger.info("Subsampled to %d / %d timesteps", len(time_indices), len(time_indices))

    plotter = OATPlotter(
        predictor=predictor,
        dataset=dataset,
        obs_station_indices=obs_station_indices,
        eval_indices=eval_indices,
        n_samples=args.n_samples,
        device="cpu",
        plot_config=_load_plot_config(dataset),
    )

    ckpt_tag = Path(args.checkpoint).parent.name
    fig = plotter.plot_pit(
        time_indices,
        title=(
            f"PIT calibration — {ckpt_tag}\n"
            f"{args.split} split  |  {len(time_indices)} timesteps  |  {args.n_samples} samples"
        ),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or str(OUT_DIR / f"{ckpt_tag}_pit_{args.split}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Saved → %s", out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
