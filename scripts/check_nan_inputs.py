"""
Diagnostic: check for NaNs and extreme values in ERA5+Meteostat batches
before and after transforms, for the temp_precip config.

Run with:
    python -m scripts.check_nan_inputs
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

NUM_BATCHES = 5
BATCH_SIZE = 4
STATS_POINTS = 256


def _check(name: str, t: torch.Tensor) -> None:
    finite = t[torch.isfinite(t)]
    n_nan = torch.isnan(t).sum().item()
    n_inf = torch.isinf(t).sum().item()
    lo = finite.min().item() if finite.numel() else float("nan")
    hi = finite.max().item() if finite.numel() else float("nan")
    flag = " *** NAN ***" if n_nan else (" *** INF ***" if n_inf else "")
    print(f"  {name:<50s} shape={str(tuple(t.shape)):<22s} "
          f"nan={n_nan:<6d} inf={n_inf:<5d} min={lo:+.4g}  max={hi:+.4g}{flag}")


def _check_batch(tag: str, batch: dict, var_names: dict | None = None) -> None:
    print(f"\n{'='*10} {tag} {'='*10}")
    for top_key in ("global_data", "observation_data", "query_data"):
        if top_key not in batch:
            continue
        for k, v in batch[top_key].items():
            if isinstance(v, torch.Tensor):
                _check(f"{top_key}/{k}", v)
                # Per-channel breakdown for global_field if it has NaN
                if k == "global_field" and torch.isnan(v).any():
                    names = var_names.get("global_field", []) if var_names else []
                    for c in range(v.shape[-3]):
                        label = names[c] if c < len(names) else str(c)
                        _check(f"  global_field[{label}]", v[..., c, :, :])


def main() -> None:
    # Compose config through Hydra so that `catalog` and settings are resolved
    root = os.path.join(os.path.dirname(__file__), "..")
    config_dir = os.path.abspath(os.path.join(root, "configs"))

    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["dataset=era5_meteostat/temp_precip"],
        )

    from oat.data_module.legacy.era5_meteostat_dataset import ERA5MeteostatDataset
    from oat.data_module.legacy.data_split import create_splits

    print("Building dataset ...")
    ds = ERA5MeteostatDataset(cfg.dataset)
    print(f"Dataset size: {len(ds)} samples")

    splits = create_splits(ds, test_days=365)
    train_ds = splits["datasets"][0]
    print(f"Train size: {len(train_ds)} samples")

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── 1. Raw data ──────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("RAW DATA (before transforms)")
    print("─" * 70)
    for i, batch in enumerate(loader):
        if i >= NUM_BATCHES:
            break
        _check_batch(f"batch {i}", batch)

    # ── 2. Build transforms ──────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Building transforms ...")
    print("─" * 70)
    fit_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    input_transform, output_transform = train_ds.get_transform(
        num_data_points=STATS_POINTS,
        dataloader=fit_loader,
        type="keep_support",  # matches oat.yaml: transform_type: keep_support
    )
    print("Transforms built successfully.")

    # ── 3. Transformed data ──────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("TRANSFORMED DATA (after input_transform)")
    print("─" * 70)
    loader2 = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    for i, batch in enumerate(loader2):
        if i >= NUM_BATCHES:
            break
        transformed = input_transform(batch)
        _check_batch(f"batch {i}", transformed)


if __name__ == "__main__":
    main()
