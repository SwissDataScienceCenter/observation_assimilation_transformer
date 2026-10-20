"""
Extract NWP variables from an anemoi dataset into a rechunked zarr.

The anemoi zarr is chunked as (1, 103, 1, 919620) — each __getitem__ read
loads a full 379 MB chunk even if only 1 variable is needed.  This script
copies the requested variables into a new zarr chunked as (1, n_vars, n_grid),
reducing per-read I/O from 379 MB to ~3.6 MB per timestep (for T_2M only).

This is a one-time operation; submit as a SLURM job for large datasets.

Usage:
    poetry run python -m scripts.extract_nwp_vars
    poetry run python -m scripts.extract_nwp_vars --vars T_2M U_10M V_10M \\
        --output <data_root>/cosmo-1e_rechunked/nwp_extracted.zarr
"""

import argparse
import time

import numpy as np
import zarr
from anemoi.datasets import open_dataset

from oat.paths import data_root

DEFAULT_INPUT = str(data_root() / "mch-co1e-an-archive-0p01-2019-2024-1h-v1-pl13.zarr")
DEFAULT_OUTPUT = str(data_root() / "cosmo-1e_rechunked" / "nwp_extracted.zarr")
DEFAULT_VARS = ["T_G", "T_2M", "U_10M", "V_10M"]  # T at 2m, U and V at 10m; adjust as needed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract NWP variables from anemoi dataset into a rechunked zarr"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to anemoi zarr")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output zarr path")
    parser.add_argument(
        "--vars", nargs="+", default=DEFAULT_VARS,
        help=f"Variable names to extract (default: {DEFAULT_VARS})"
    )
    parser.add_argument(
        "--batch", type=int, default=64,
        help="Timesteps to write per iteration (default: 64)"
    )
    args = parser.parse_args()

    print(f"Opening anemoi dataset: {args.input}", flush=True)
    ds = open_dataset(args.input)

    missing = [v for v in args.vars if v not in ds.name_to_index]
    if missing:
        raise KeyError(f"Variables not found in dataset: {missing}. "
                       f"Available: {sorted(ds.name_to_index.keys())}")

    var_idx = [ds.name_to_index[v] for v in args.vars]
    n_times, full_num_vars, _, n_grid = ds.data.shape   # (T, 103, 1, G)
    n_vars = len(var_idx)
    full_chunk_mb = full_num_vars * n_grid * 4 / 1e6  # 103 variables, float32, per timestep
    chunk_mb = n_vars * n_grid * 4 / 1e6
    

    print(f"Variables : {args.vars}  (indices {var_idx})", flush=True)
    print(f"Shape     : ({n_times}, {n_vars}, {n_grid})", flush=True)
    print(f"Chunk size: {chunk_mb:.1f} MB  (was {full_chunk_mb:.1f} MB, {full_chunk_mb/chunk_mb:.0f}x reduction)", flush=True)
    print(f"Output    : {args.output}", flush=True)

    store = zarr.open(
        args.output,
        mode="w",
        shape=(n_times, n_vars, n_grid),
        chunks=(1, n_vars, n_grid),
        dtype=np.float32,
    )
    store.attrs["variables"] = args.vars

    t0 = time.perf_counter()
    for start in range(0, n_times, args.batch):
        end = min(start + args.batch, n_times)
        # ds.data[start:end] shape: (batch, 103, 1, n_grid)
        batch_data = ds.data[start:end][:, var_idx, 0, :]  # (batch, n_vars, n_grid)
        store[start:end] = batch_data
        elapsed = time.perf_counter() - t0
        rate = (end) / elapsed
        eta = (n_times - end) / rate if rate > 0 else 0
        print(
            f"  {end}/{n_times}  ({100*end/n_times:.1f}%)  "
            f"{elapsed:.0f}s elapsed  ETA {eta:.0f}s",
            flush=True,
        )

    print(f"\nDone in {time.perf_counter() - t0:.0f}s. Output: {args.output}", flush=True)


if __name__ == "__main__":
    main()
