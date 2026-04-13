"""
Benchmark DataLoader throughput vs. number of workers.

Loads dataset config from configs/dataset/cosmo_peakweather.yaml (via OmegaConf).

Usage:
    python -m scripts.benchmark_dataloader
    python -m scripts.benchmark_dataloader --workers 0 1 2 4 8 --batches 20 --batch-size 32
    python -m scripts.benchmark_dataloader --start-date 2022-01-01 --end-date 2022-01-31
"""

import argparse
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from oat.data_module.cosmo_peakweather_dataset import CosmoPeakWeatherDataset

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "dataset" / "cosmo_peakweather.yaml"


def benchmark(dataset: CosmoPeakWeatherDataset, num_workers: int, batch_size: int,
              num_batches: int, prefetch_factor: int = 2) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        shuffle=True,
    )

    batch_times = []
    loader_iter = iter(loader)

    # Warmup: one batch to trigger worker startup / zarr store init
    print(f"  workers={num_workers}: warming up...", flush=True)
    t0 = time.perf_counter()
    try:
        next(loader_iter)
    except StopIteration:
        return {}
    warmup_time = time.perf_counter() - t0
    print(f"  workers={num_workers}: warmup {warmup_time:.2f}s", flush=True)

    # Timed batches
    for i in range(num_batches):
        t0 = time.perf_counter()
        try:
            next(loader_iter)
        except StopIteration:
            break
        elapsed = time.perf_counter() - t0
        batch_times.append(elapsed)
        print(f"  workers={num_workers}: batch {i+1}/{num_batches}  {elapsed:.3f}s", flush=True)

    if not batch_times:
        return {}

    arr = np.array(batch_times)
    return {
        "num_workers": num_workers,
        "warmup_s": warmup_time,
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "p95_s": float(np.percentile(arr, 95)),
        "max_s": float(arr.max()),
        "throughput_samples_per_s": batch_size / float(arr.mean()),
        "n_batches": len(arr),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DataLoader workers vs. throughput")
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 1, 2, 4, 8],
                        help="num_workers values to test (default: 0 1 2 4 8)")
    parser.add_argument("--batches", type=int, default=10,
                        help="Timed batches per run, after warmup (default: 10)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prefetch", type=int, default=2)
    args = parser.parse_args()

    cfg = OmegaConf.load(CONFIG_PATH)


    print(f"Config loaded from {CONFIG_PATH}", flush=True)
    print("Loading dataset...", flush=True)
    t0 = time.perf_counter()
    dataset = CosmoPeakWeatherDataset(cfg)
    print(f"Dataset ready in {time.perf_counter() - t0:.1f}s  ({len(dataset)} samples)\n", flush=True)

    results = []
    for nw in args.workers:
        print(f"\n--- num_workers={nw} ---", flush=True)
        r = benchmark(dataset, nw, args.batch_size, args.batches, args.prefetch)
        if r:
            results.append(r)

    print("\n" + "=" * 72)
    print(f"{'workers':>8}  {'warmup':>8}  {'mean':>7}  {'median':>7}  {'p95':>7}  {'max':>7}  {'samples/s':>10}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['num_workers']:>8}  "
            f"{r['warmup_s']:>7.2f}s  "
            f"{r['mean_s']:>6.3f}s  "
            f"{r['median_s']:>6.3f}s  "
            f"{r['p95_s']:>6.3f}s  "
            f"{r['max_s']:>6.3f}s  "
            f"{r['throughput_samples_per_s']:>9.1f}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
