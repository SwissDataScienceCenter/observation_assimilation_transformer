"""
Diagnostic: check for NaNs and extreme values in ERA5+Meteostat batches
before and after transforms, for the temp_precip config.

Run with:
    python -m scripts.check_nan_inputs
"""
import logging
import os

import torch

NUM_BATCHES = 5


def _check(name: str, t: torch.Tensor) -> None:
    finite = t[torch.isfinite(t)]
    n_nan = torch.isnan(t).sum().item()
    n_inf = torch.isinf(t).sum().item()
    lo = finite.min().item() if finite.numel() else float("nan")
    hi = finite.max().item() if finite.numel() else float("nan")
    flag = " *** NAN ***" if n_nan else (" *** INF ***" if n_inf else "")
    print(f"  {name:<50s} shape={str(tuple(t.shape)):<22s} "
          f"nan={n_nan:<6d} inf={n_inf:<5d} min={lo:+.4g}  max={hi:+.4g}{flag}")


def _check_batch(tag: str, batch: dict) -> None:
    print(f"\n{'='*10} {tag} {'='*10}")
    for top_key in ("global_data", "observation_data", "query_data"):
        if top_key not in batch or "fields" not in batch[top_key]:
            continue
        for field_name, field in batch[top_key]["fields"].items():
            _check(f"{top_key}/{field_name}.values", field.values)
            if field.mask is not None:
                _check(f"{top_key}/{field_name}.mask", field.mask)
            # Per-channel breakdown if this field has NaN.
            if torch.isnan(field.values).any():
                for c, var_name in enumerate(field.var_names):
                    _check(
                        f"  {top_key}/{field_name}[{var_name}]",
                        field.values[..., c],
                    )


def main() -> None:
    # Compose config through Hydra so that `catalog` and settings are resolved
    root = os.path.join(os.path.dirname(__file__), "..")
    config_dir = os.path.abspath(os.path.join(root, "configs"))

    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset=era5_meteostat/temp_precip",
                "dataloader.train.batch_size=4",
                "dataloader.train.num_workers=0",
                "dataloader.train.persistent_workers=false",
                "dataloader.train.prefetch_factor=null",
            ],
        )

    from oat.data_module.setup import setup_data_and_transform

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger(__name__)

    print("Building dataset and dataloaders through the live field setup path ...")
    dataset, n_train, train_loader, _val_loader, _test_loader, input_transform, _ = (
        setup_data_and_transform(cfg, logger)
    )
    print(f"Train size: {n_train} samples")

    # ── 1. Raw data ──────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("RAW DATA (before transforms)")
    print("─" * 70)
    for i, batch in enumerate(train_loader):
        if i >= NUM_BATCHES:
            break
        _check_batch(f"batch {i}", batch)

    # ── 2. Transformed data ──────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("TRANSFORMED DATA (after input_transform)")
    print("─" * 70)
    for i, batch in enumerate(train_loader):
        if i >= NUM_BATCHES:
            break
        transformed = input_transform(batch)
        _check_batch(f"batch {i}", transformed)


if __name__ == "__main__":
    main()
