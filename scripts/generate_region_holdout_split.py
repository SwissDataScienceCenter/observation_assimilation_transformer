"""Generate a region-holdout split config for spatial evaluation.

Split stations into train, val, test-context, and test-eval groups.
Use a lat/lon box to pick the test region. Split each side of the box
by a random, seeded draw.

Usage
-----
python -m scripts.generate_region_holdout_split [options]

Options
-------
--lat-min FLOAT              Minimum latitude (default: 44)
--lat-max FLOAT              Maximum latitude (default: 48)
--lon-min FLOAT              Minimum longitude (default: 5)
--lon-max FLOAT              Maximum longitude (default: 10.5)
--context-fraction FLOAT     Fraction of the box stations for test-context
                              (default: 0.8)
--val-station-fraction FLOAT Fraction of the out-of-box stations for val
                              (default: 0.2)
--seed INT                   Random seed (default: 42)
--station-meta-path PATH     Path to the station metadata file
--obs-path PATH              Path to the observation file
--output PATH                Output config path
                              (default: configs/split/spatial_region_holdout.yaml)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def load_station_coords(station_meta_path: str, obs_path: str) -> pd.DataFrame:
    """Load the station coordinates used by the live dataset.

    Apply the same station filter as ``MeteostatObservations.__init__``
    in ``era5_meteostat_dataset.py``. Keep only stations with full geo
    metadata. Keep only stations that appear in the observation file.

    Parameters
    ----------
    station_meta_path : str
        Path to the station metadata file, in parquet format.
    obs_path : str
        Path to the observation file, in parquet format.

    Returns
    -------
    pd.DataFrame
        Station ID (index) -> latitude, longitude columns. Keep the
        country column too, when the metadata file has one.
    """
    meta = pd.read_parquet(station_meta_path)
    if "station" in meta.columns:
        meta = meta.set_index("station")

    keep_cols = ["latitude", "longitude"]
    if "country" in meta.columns:
        keep_cols.append("country")

    filtered_meta = meta.dropna(subset=["latitude", "longitude", "elevation"])[
        keep_cols
    ].copy()
    filtered_meta.index = filtered_meta.index.astype(str)

    obs = pd.read_parquet(obs_path)
    obs["station"] = obs["station"].astype(str)
    obs = obs[obs["station"].isin(set(filtered_meta.index))]
    station_ids = sorted(obs["station"].unique())

    return filtered_meta.loc[station_ids]


def compute_region_holdout_split(
    station_coords: pd.DataFrame,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    context_fraction: float,
    val_station_fraction: float,
    seed: int,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Compute a region-holdout station split.

    Find the stations inside the lat/lon box. Split them into a
    test-context group and a test-eval group.
    Find the stations outside the box. Split them into a train group
    and a val group.

    Parameters
    ----------
    station_coords : pd.DataFrame
        Station ID (index) -> latitude, longitude columns.
    lat_min : float
        Minimum latitude. This bound is inclusive.
    lat_max : float
        Maximum latitude. This bound is inclusive.
    lon_min : float
        Minimum longitude. This bound is inclusive.
    lon_max : float
        Maximum longitude. This bound is inclusive.
    context_fraction : float
        Fraction of the in-box stations to put in the test-context
        group. Put the rest in the test-eval group.
    val_station_fraction : float
        Fraction of the out-of-box stations to put in the val group.
        Put the rest in the train group.
    seed : int
        Random seed. Use it to shuffle stations before the split.

    Returns
    -------
    tuple[list[str], list[str], list[str], list[str]]
        Four sorted station-ID lists: train_ids, val_ids,
        test_context_ids, test_eval_ids.

    Raises
    ------
    ValueError
        Raise this error when the box selects no stations.
        Raise this error when context_fraction leaves a group empty.
        Raise this error when val_station_fraction leaves a group empty.
    """
    in_box = (
        (station_coords["latitude"] >= lat_min)
        & (station_coords["latitude"] <= lat_max)
        & (station_coords["longitude"] >= lon_min)
        & (station_coords["longitude"] <= lon_max)
    )
    holdout_ids = sorted(station_coords.index[in_box].tolist())
    other_ids = sorted(station_coords.index[~in_box].tolist())

    if not holdout_ids:
        raise ValueError("no stations in the region box")

    rng = np.random.RandomState(seed)

    num_holdout = len(holdout_ids)
    num_context = int(num_holdout * context_fraction)
    num_eval = num_holdout - num_context
    if num_context == 0 or num_eval == 0:
        raise ValueError("context_fraction would leave a group empty")

    holdout_perm = rng.permutation(holdout_ids)
    test_context_ids = sorted(holdout_perm[:num_context].tolist())
    test_eval_ids = sorted(holdout_perm[num_context:].tolist())

    num_other = len(other_ids)
    num_val = int(num_other * val_station_fraction)
    num_train = num_other - num_val
    if num_val == 0 or num_train == 0:
        raise ValueError("val_station_fraction would leave a group empty")

    other_perm = rng.permutation(other_ids)
    val_ids = sorted(other_perm[:num_val].tolist())
    train_ids = sorted(other_perm[num_val:].tolist())

    return train_ids, val_ids, test_context_ids, test_eval_ids


def render_config(
    train_ids: list[str],
    val_ids: list[str],
    test_context_ids: list[str],
    test_eval_ids: list[str],
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    context_fraction: float,
    val_station_fraction: float,
    seed: int,
    station_meta_path: str,
    obs_path: str,
) -> str:
    """Render the region-holdout split config as YAML text.

    Add a header comment. Document the region bounds, the split
    fractions, the seed, and the source file paths.

    Parameters
    ----------
    train_ids : list[str]
        Train station IDs.
    val_ids : list[str]
        Val station IDs.
    test_context_ids : list[str]
        Test-context station IDs.
    test_eval_ids : list[str]
        Test-eval station IDs.
    lat_min : float
        Minimum latitude of the region box.
    lat_max : float
        Maximum latitude of the region box.
    lon_min : float
        Minimum longitude of the region box.
    lon_max : float
        Maximum longitude of the region box.
    context_fraction : float
        Fraction used to build the test-context group.
    val_station_fraction : float
        Fraction used to build the val group.
    seed : int
        Random seed used for the split.
    station_meta_path : str
        Path to the station metadata file (for the header comment).
    obs_path : str
        Path to the observation file (for the header comment).

    Returns
    -------
    str
        Complete YAML config text.
    """
    lines = [
        "# Region-holdout split config",
        f"# Generated: {datetime.now().isoformat()}",
        f"# Region bounds: lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]",
        f"# context_fraction: {context_fraction}",
        f"# val_station_fraction: {val_station_fraction}",
        f"# seed: {seed}",
        f"# source: {station_meta_path}, {obs_path}",
        "#",
        "defaults:",
        "  - default",
        "",
        "train_station_indices:",
    ]
    for sid in train_ids:
        lines.append(f'  - "{sid}"')

    lines.append("val_station_indices:")
    for sid in val_ids:
        lines.append(f'  - "{sid}"')

    lines.append("test_station_indices:")
    for sid in test_eval_ids:
        lines.append(f'  - "{sid}"')

    lines.append("test_context_station_indices:")
    for sid in test_context_ids:
        lines.append(f'  - "{sid}"')

    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--lat-min", type=float, default=44, help="Minimum latitude (default: 44)"
    )
    p.add_argument(
        "--lat-max", type=float, default=48, help="Maximum latitude (default: 48)"
    )
    p.add_argument(
        "--lon-min", type=float, default=5, help="Minimum longitude (default: 5)"
    )
    p.add_argument(
        "--lon-max", type=float, default=10.5, help="Maximum longitude (default: 10.5)"
    )
    p.add_argument(
        "--context-fraction",
        type=float,
        default=0.8,
        help="Fraction of the box stations for test-context (default: 0.8)",
    )
    p.add_argument(
        "--val-station-fraction",
        type=float,
        default=0.2,
        help="Fraction of the out-of-box stations for val (default: 0.2)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--station-meta-path",
        type=str,
        default=(
            "/scratch/mch/cdonner/meteostats_dataset/data/processed/"
            "station_quality.parquet"
        ),
        help="Path to the station metadata file",
    )
    p.add_argument(
        "--obs-path",
        type=str,
        default=(
            "/scratch/mch/cdonner/meteostats_dataset/data/processed/"
            "hourly_2019_2025_temp_prcp_wspd_wpgt_rhum.parquet"
        ),
        help="Path to the observation file",
    )
    p.add_argument(
        "--output",
        type=str,
        default="configs/split/spatial_region_holdout.yaml",
        help="Output config path",
    )
    args = p.parse_args()

    coords = load_station_coords(args.station_meta_path, args.obs_path)

    train_ids, val_ids, ctx_ids, eval_ids = compute_region_holdout_split(
        coords,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        context_fraction=args.context_fraction,
        val_station_fraction=args.val_station_fraction,
        seed=args.seed,
    )

    config_content = render_config(
        train_ids,
        val_ids,
        ctx_ids,
        eval_ids,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        context_fraction=args.context_fraction,
        val_station_fraction=args.val_station_fraction,
        seed=args.seed,
        station_meta_path=args.station_meta_path,
        obs_path=args.obs_path,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(config_content)

    print(
        f"wrote {output_path}: {len(train_ids)} train, {len(val_ids)} val, "
        f"{len(ctx_ids)} test-context, {len(eval_ids)} test-eval stations"
    )


if __name__ == "__main__":
    main()
