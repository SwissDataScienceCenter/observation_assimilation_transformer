"""Aggregate performance metrics over the test (or val) split.

Runs OATPlotter.compute_station_metrics over all eval stations and timesteps in
the chosen split, prints a global summary, and saves a per-station CSV.

Usage
-----
python -m scripts.evaluate_test_set <checkpoint> [options]

Options
-------
--split {val,test}    Split to evaluate (default: test)
--baseline            Include NWP interpolator as reference (enables CRPSS)
--n-samples N         Predictive samples per forward pass (default: 100)
--max-timesteps M     Evaluate only the first M timesteps (quick smoke-test)
--device cpu|cuda     Torch device (default: cpu)
--out PATH            Output CSV (default: data/results/<tag>_metrics_<split>.csv)
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

from oat.data_module.setup import setup_data_and_transform
from oat.predictor_module.checkpoint_utils import (
    _DEFAULT_REGISTRY,
    load_checkpoint,
    resolve_checkpoint,
)
from scripts.plotter_utils import (
    is_field_path,
    maybe_build_nwp_baseline,
    select_plotter_cls,
)

OUT_DIR = Path(__file__).parents[1] / "data/results"

# Weightless NWP-interpolation baseline: no checkpoint of its own, so it borrows
# the data config from any sibling checkpoint of the same dataset.
INTERPOLATOR_MODEL = "interpolator"


def _sibling_checkpoint(dataset: str) -> str:
    """Return any non-null checkpoint path for ``dataset`` (to borrow its cfg)."""
    registry = OmegaConf.load(_DEFAULT_REGISTRY)
    models = registry.get(dataset) or {}
    for model, path in models.items():
        if model != INTERPOLATOR_MODEL and path is not None:
            return str(path)
    raise ValueError(
        f"No sibling checkpoint to borrow a data config from for the "
        f"interpolator on dataset={dataset!r}."
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate test-set metrics for an OAT checkpoint."
    )
    p.add_argument("checkpoint", nargs="?", default=None, help="Path to .ckpt file")
    p.add_argument(
        "--dataset", default=None, help="Dataset key in configs/checkpoints.yaml"
    )
    p.add_argument(
        "--model", default=None, help="Model key in configs/checkpoints.yaml"
    )
    p.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Which split to evaluate (default: test)",
    )
    p.add_argument(
        "--baseline",
        action="store_true",
        help="Include NWP interpolator as reference predictor (enables CRPSS)",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=100,
        metavar="N",
        help="Predictive samples per forward pass (default: 100)",
    )
    p.add_argument(
        "--max-timesteps",
        type=int,
        default=None,
        metavar="M",
        help="Evaluate only the first M timesteps",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Torch device for inference (default: cpu)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output CSV path",
    )
    p.add_argument(
        "--batch-timesteps",
        type=int,
        default=32,
        metavar="B",
        help="Timesteps per inference batch (default: 32)",
    )
    p.add_argument(
        "--long",
        action="store_true",
        help="Write a per-(station, timestep) Parquet dump for stratification "
        "plots instead of the aggregate CSV.",
    )
    return p.parse_args()


def _print_summary(metrics: dict, var_names: list[str], logger: logging.Logger) -> None:
    # ponytail: plain logging lines — no tabulate dependency
    col_w = max(12, max(len(v) for v in var_names) + 2)
    header = f"{'metric':<14}" + "".join(f"{v:>{col_w}}" for v in var_names)
    logger.info(header)
    logger.info("-" * len(header))
    for key in ("crps", "mae", "crpss"):
        global_key = f"{key}_global"
        if global_key not in metrics:
            continue
        per_var = torch.nanmean(
            metrics[key], dim=0
        )  # average over stations, ignoring NaN → [V]
        vals = "".join(
            f"{per_var[i].item():>{col_w}.4f}" for i in range(len(var_names))
        )
        logger.info(f"{key:<14}{vals}   (global {metrics[global_key].item():.4f})")


def _aggregate_station_time(tm: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Derive per-station + global CRPS/MAE from the per-(time, station) dump.

    The mask is binary, so ``compute_station_metrics``' count-weighted mean over
    time is exactly a ``nanmean`` over the time axis of the long tensors — this
    reproduces the aggregate without a second inference pass. CRPSS is not
    derivable here (it needs a joint model+baseline pass) and is omitted.
    """
    crps, mae = tm["crps"], tm["mae"]  # each [T, S, V], NaN where masked
    return {
        "crps": torch.nanmean(crps, dim=0),  # [S, V]
        "mae": torch.nanmean(mae, dim=0),  # [S, V]
        "crps_global": torch.nanmean(crps),
        "mae_global": torch.nanmean(mae),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

    args = _parse_args()

    is_interpolator = args.model == INTERPOLATOR_MODEL

    if is_interpolator:
        # Borrow the data config from a sibling checkpoint; build predictor below.
        cfg = OmegaConf.load(
            Path(_sibling_checkpoint(args.dataset)).parent / "config.yaml"
        )
        predictor = None
    else:
        if args.checkpoint is None:
            if args.dataset is None or args.model is None:
                logging.getLogger(__name__).error(
                    "Provide either a checkpoint path or both --dataset and --model."
                )
                raise SystemExit(2)
            args.checkpoint = resolve_checkpoint(args.dataset, args.model)
        predictor, cfg = load_checkpoint(None, args.checkpoint, map_location="cpu")
        predictor = predictor.eval()

    cfg.dataset.data_points_for_stats = 0

    dataset, _, _, val_loader, test_loader, *_ = setup_data_and_transform(cfg, logger)

    if is_interpolator:
        if is_field_path(dataset):
            logger.warning(
                "--model interpolator is not yet supported on the Field data "
                "path; skipping."
            )
            raise SystemExit(0)
        from oat.model_module.baseline import NWPInterpolatorConfig
        from oat.predictor_module.baseline_predictor import NWPInterpolatorPredictor

        nwp_config = NWPInterpolatorConfig(field_extent=dataset.nwp_extent)
        predictor = NWPInterpolatorPredictor(
            config=nwp_config,
            output_var_names=dataset.config.obs_vars,
            input_transform=dataset.get_baseline_pre_transform(),
        ).eval()

    split_loader = val_loader if args.split == "val" else test_loader
    split_ds = split_loader.dataset  # EvalDataset

    obs_indices = list(map(int, split_ds.observed_indices))
    eval_indices = list(map(int, split_ds.eval_indices))

    if split_ds._time_indices is not None:
        time_indices = list(map(int, split_ds._time_indices))
    else:
        time_indices = list(range(len(dataset)))

    if args.max_timesteps is not None:
        time_indices = time_indices[: args.max_timesteps]

    logger.info(
        "split=%s  timesteps=%d  obs_stations=%d  eval_stations=%d",
        args.split,
        len(time_indices),
        len(obs_indices),
        len(eval_indices),
    )

    PlotterCls = select_plotter_cls(dataset)
    plotter = PlotterCls(
        predictor=predictor,
        dataset=dataset,
        obs_station_indices=obs_indices,
        eval_indices=eval_indices,
        n_samples=args.n_samples,
        device=args.device,
    )

    nwp_predictor = None
    if args.baseline and not is_interpolator:
        nwp_predictor = maybe_build_nwp_baseline(
            dataset, dataset.config.obs_vars, logger
        )

    var_names = list(predictor.var_to_indx.keys())

    tm: dict[str, torch.Tensor] | None = None
    if args.long:
        from oat.plotting_module.stratification import (
            build_long_dataframe,
            collect_station_covariates,
        )

        tm = plotter.compute_station_time_metrics(
            time_indices=time_indices,
            eval_station_indices=eval_indices,
            batch_size=args.batch_timesteps,
        )
        covariates = collect_station_covariates(dataset, eval_indices, obs_indices)
        df = build_long_dataframe(
            crps=tm["crps"].numpy(),
            mae=tm["mae"].numpy(),
            times=[dataset.times[t] for t in time_indices],
            covariates=covariates,
            var_names=var_names,
        )
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_tag = (
            f"{INTERPOLATOR_MODEL}_{args.dataset}"
            if is_interpolator
            else Path(args.checkpoint).parent.name
        )
        long_path = (
            Path(args.out)
            if args.out
            else OUT_DIR / f"{ckpt_tag}_long_{args.split}.parquet"
        )
        df.to_parquet(long_path, index=False)
        logger.info("Saved long dump (%d rows) → %s", len(df), long_path)

    if tm is not None and nwp_predictor is None:
        # Aggregate exactly from the long dump — no extra inference pass.
        metrics = _aggregate_station_time(tm)
    else:
        # No long dump, or CRPSS wanted (needs a joint model+baseline pass).
        metrics = plotter.compute_station_metrics(
            time_indices=time_indices,
            eval_station_indices=eval_indices,
            reference=nwp_predictor,
            batch_size=args.batch_timesteps,
        )

    _print_summary(metrics, var_names, logger)

    # Build per-station CSV
    coords = dataset.station_coords[eval_indices].numpy()  # [S, 2]
    rows: dict = {
        "station_x": coords[:, 0].tolist(),
        "station_y": coords[:, 1].tolist(),
    }
    for key in ("crps", "mae", "crpss"):
        if key not in metrics:
            continue
        tensor = metrics[key]  # [S, V]
        for vi, var in enumerate(var_names):
            rows[f"{key}_{var}"] = tensor[:, vi].tolist()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_tag = (
        f"{INTERPOLATOR_MODEL}_{args.dataset}"
        if is_interpolator
        else Path(args.checkpoint).parent.name
    )
    # In --long mode, args.out names the Parquet dump; auto-name the CSV to avoid a clash.
    out_path = (
        Path(args.out)
        if args.out and not args.long
        else OUT_DIR / f"{ckpt_tag}_metrics_{args.split}.csv"
    )
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info("Saved → %s", out_path)

    summary: dict = {}
    for key in ("crps", "mae", "crpss"):
        global_key = f"{key}_global"
        if global_key not in metrics:
            continue
        summary[global_key] = metrics[global_key].item()
        per_var = {
            var: torch.nanmean(metrics[key][:, vi]).item()
            for vi, var in enumerate(var_names)
        }
        summary[key] = per_var
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved summary → %s", summary_path)


if __name__ == "__main__":
    main()
