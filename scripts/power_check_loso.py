"""Model-free power-check for the LOSO near-station calibration diagnostic.

Counts, over the test split, how many *scorable* target-station x hour observations
have a valid *context*-station neighbour within distance bins (<2, 2-5, 5-10 km),
under the deployment geometry: targets = test stations, context = train+val stations
(exactly how the checkpoint was trained/evaluated). No model weights are loaded --
this is pure geometry + observation-validity masks. It decides whether a near-station
PIT/rank-histogram calibration diagnostic can be powered at all.

Usage:
    python -m scripts.power_check_loso --checkpoint <path-to-.ckpt>

The checkpoint is used only to locate its sibling ``config.yaml`` (dataset paths,
obs_vars, and the exact train/val/test station split).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

from oat.data_module.legacy.data_split import create_splits
from oat.data_module.setup import _instantiate_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("power_check_loso")

EARTH_RADIUS_KM = 6371.0088


def latlon_to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Map (lat, lon) in degrees to 3-D points on a sphere of radius EARTH_RADIUS_KM."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    x = EARTH_RADIUS_KM * np.cos(lat) * np.cos(lon)
    y = EARTH_RADIUS_KM * np.cos(lat) * np.sin(lon)
    z = EARTH_RADIUS_KM * np.sin(lat)
    return np.column_stack([x, y, z])


def chord_len(arc_km: float) -> float:
    """Straight-line (chord) distance for a great-circle arc of ``arc_km`` km.

    Used as the cKDTree query radius on the 3-D embedding; for <10 km the chord
    under-approximates the arc by <1e-5, so we query at the chord for the largest
    arc bin and then filter on the exact haversine distance.
    """
    return 2.0 * EARTH_RADIUS_KM * np.sin(arc_km / (2.0 * EARTH_RADIUS_KM))


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km between (lat1,lon1) and arrays (lat2,lon2)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def nearest_valid_distance(
    tgt_lat, tgt_lon, ctx_lat, ctx_lon, mask_ctx, max_km: float
) -> np.ndarray:
    """Distance (km) from each target to its nearest *valid* context station, per hour.

    Returns an array of shape [T, n_target]; entries are inf where no valid context
    station lies within ``max_km`` of the target at that hour. Only context stations
    within ``max_km`` of a target ever matter, so we restrict to each target's
    neighbourhood (sparse) rather than forming a dense [n_target, n_context] product.
    """
    n_t = tgt_lat.shape[0]
    T = mask_ctx.shape[0]
    tree = cKDTree(latlon_to_xyz(ctx_lat, ctx_lon))
    ball = tree.query_ball_point(latlon_to_xyz(tgt_lat, tgt_lon), r=chord_len(max_km))

    nearest = np.full((T, n_t), np.inf, dtype=np.float32)
    for i, nb in enumerate(ball):
        if not nb:
            continue
        nb = np.asarray(nb, dtype=int)
        d = haversine_km(tgt_lat[i], tgt_lon[i], ctx_lat[nb], ctx_lon[nb])
        keep = d <= max_km  # drop the chord/arc slack beyond the true arc bin
        nb, d = nb[keep], d[keep]
        if nb.size == 0:
            continue
        order = np.argsort(d)
        nb, d = nb[order], d[order]
        valid = mask_ctx[:, nb]  # [T, k], neighbours in ascending distance
        has = valid.any(axis=1)
        first = valid.argmax(axis=1)  # index of nearest valid neighbour (0 if none)
        nearest[:, i] = np.where(has, d[first], np.inf)
    return nearest


def bin_counts(nearest: np.ndarray, scorable: np.ndarray, edges) -> dict:
    """Count scorable target-hours whose nearest valid context falls in each bin."""
    out = {}
    lo = 0.0
    for hi in edges:
        label = f"<{hi:g}km" if lo == 0.0 else f"{lo:g}-{hi:g}km"
        out[label] = int((scorable & (nearest >= lo) & (nearest < hi)).sum())
        lo = hi
    out[f">={edges[-1]:g}km/none"] = int((scorable & (nearest >= edges[-1])).sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="Path to the .ckpt (uses its sibling config.yaml).")
    ap.add_argument("--distance-bins", default="2,5,10", help="Comma-separated bin edges in km.")
    ap.add_argument("--out-dir", default="data/results")
    ap.add_argument("--fig-dir", default="data/figures")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    edges = [float(x) for x in args.distance_bins.split(",")]
    max_km = edges[-1]

    config_path = Path(args.checkpoint).parent / "config.yaml"
    cfg = OmegaConf.load(config_path)
    logger.info("Loaded config from %s (dataset_type=%s)", config_path, cfg.dataset.dataset_type)

    logger.info("Instantiating dataset (no model weights loaded)...")
    dataset = _instantiate_dataset(cfg)
    splits = create_splits(dataset, **cfg.split, logger=logger)
    train_idx, val_idx, test_idx = splits["station_indices"]
    test_time_indices = np.asarray(splits["test_time_indices"], dtype=int)

    context_idx = np.asarray(sorted(train_idx) + sorted(val_idx), dtype=int)
    target_idx = np.asarray(sorted(test_idx), dtype=int)
    logger.info(
        "Deployment geometry: %d target (test) stations, %d context (train+val) stations, %d test hours.",
        target_idx.size, context_idx.size, test_time_indices.size,
    )

    coords = dataset.station_coords.numpy()  # [N, 2] = (lon, lat) degrees
    tgt_lon, tgt_lat = coords[target_idx, 0], coords[target_idx, 1]
    ctx_lon, ctx_lat = coords[context_idx, 0], coords[context_idx, 1]

    # obs_mask_tensor is [len(times), N, V]; True == valid observation.
    mask_all = dataset.obs_mask_tensor[test_time_indices].numpy().astype(bool)  # [T, N, V]
    obs_vars = list(cfg.dataset.obs_vars)

    results = {
        "checkpoint": str(args.checkpoint),
        "geometry": "deployment (targets=test, context=train+val)",
        "n_target_stations": int(target_idx.size),
        "n_context_stations": int(context_idx.size),
        "n_test_hours": int(test_time_indices.size),
        "distance_bins_km": edges,
        "decision_threshold": {"lt10_aggregate": 300, "per_fine_bin": 100},
        "per_variable": {},
    }

    hist_edges = np.arange(0.0, max_km + 0.5, 0.5)
    hists = {}

    for v, var in enumerate(obs_vars):
        mask_ctx = mask_all[:, context_idx, v]  # [T, n_context]
        mask_tgt = mask_all[:, target_idx, v]   # [T, n_target]
        nearest = nearest_valid_distance(tgt_lat, tgt_lon, ctx_lat, ctx_lon, mask_ctx, max_km)

        counts = bin_counts(nearest, mask_tgt, edges)
        total = int(mask_tgt.sum())
        lt_max = int((mask_tgt & (nearest < max_km)).sum())
        assert sum(counts.values()) == total, "bin counts must partition scorable target-hours"

        hists[var] = np.histogram(nearest[mask_tgt & np.isfinite(nearest)], bins=hist_edges)[0].tolist()
        fine_bins = {k: c for k, c in counts.items() if not k.endswith("none")}
        powered = lt_max >= results["decision_threshold"]["lt10_aggregate"] and all(
            c >= results["decision_threshold"]["per_fine_bin"] for c in fine_bins.values()
        )
        results["per_variable"][var] = {
            "total_scorable_target_hours": total,
            f"scorable_within_{max_km:g}km": lt_max,
            "bin_counts": counts,
            "powered": bool(powered),
        }

        logger.info("── %s ──────────────────────────────", var)
        logger.info("scorable target-hours: %d   within %g km: %d", total, max_km, lt_max)
        for k, c in counts.items():
            logger.info("  %-14s %d", k, c)
        logger.info("  POWERED: %s", powered)

    out_dir, fig_dir = Path(args.out_dir), Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "power_check_loso.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", out_path)

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        centers = (hist_edges[:-1] + hist_edges[1:]) / 2
        for var in obs_vars:
            ax.step(centers, hists[var], where="mid", label=var)
        ax.set_xlabel("distance to nearest valid context station (km)")
        ax.set_ylabel("scorable target-hours")
        ax.set_title("Near-station power (deployment geometry)")
        ax.legend()
        fig.tight_layout()
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / "power_check_loso_distance_hist.png"
        fig.savefig(fig_path, dpi=120)
        logger.info("Wrote %s", fig_path)


if __name__ == "__main__":
    main()
