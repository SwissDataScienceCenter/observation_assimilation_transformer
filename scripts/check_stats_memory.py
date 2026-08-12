"""
Check whether RunningFeatureStats.update and pre_transform leak memory over many
iterations. Runs without real dataset files — uses synthetic tensors that mimic
the shapes produced by CosmoPeakWeatherDataset.__getitem__.
"""

import gc
import tracemalloc

import torch

from oat.data_module.statistics import RunningFeatureStats
from oat.data_module.transform import LogTransform, NestedTransform

# Mimic the shapes used in _collect_stats:
#   global_field  -> (C_nwp + C_topo, H, W)  e.g. (8, 710, 710)  feature_dim=-3
#   obs           -> (N_stations, C_obs)       e.g. (100, 4)       feature_dim=-1
#   topo          -> (N_stations, C_topo, H, W) e.g. (100, 5, 32, 32) feature_dim=-3
#   geo           -> (N_stations, 3)            e.g. (100, 3)       feature_dim=-1

CONFIGS = [
    dict(name="global_field", shape=(8, 710, 710), num_features=8, feature_dim=-3),
    dict(name="obs",          shape=(100, 4),       num_features=4, feature_dim=-1),
    dict(name="topo",         shape=(100, 5, 32, 32), num_features=5, feature_dim=-3),
    dict(name="geo",          shape=(100, 3),       num_features=3, feature_dim=-1),
]

N_ITERS = 200

# Pre-transform matching the 'real' path in get_transform():
# log-transform obs indices that have non-negative lower bound (e.g. indices [1, 2] out of 4)
LOG_INDICES = [1, 2]
PRE_TRANSFORM = NestedTransform({
    'observation_data': {
        'obs': LogTransform(apply_to=LOG_INDICES, feature_dim=-1, offset=1e-6)
    }
})


def make_batch() -> dict:
    obs = torch.abs(torch.randn(100, 4))  # non-negative so log is valid
    obs[obs < 0.01] = float("nan")
    return {
        'global_data': {
            'global_field': torch.randn(8, 710, 710),
            'time_emb': torch.randn(4),
        },
        'observation_data': {
            'obs': obs,
            'topo': torch.randn(100, 5, 32, 32),
            'geo': torch.randn(100, 3),
            'coords': torch.randn(100, 2),
            'mask_obs': torch.ones(100, 4),
        },
    }


def run_check() -> None:
    stats = {
        cfg["name"]: RunningFeatureStats(
            num_features=cfg["num_features"],
            feature_dim=cfg["feature_dim"],
        )
        for cfg in CONFIGS
    }

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    for i in range(N_ITERS):
        batch = make_batch()
        transformed = PRE_TRANSFORM(batch)

        # Mimic gather_stats
        stats["global_field"].update(transformed['global_data']['global_field'])
        stats["obs"].update(transformed['observation_data']['obs'])
        stats["topo"].update(transformed['observation_data']['topo'])
        stats["geo"].update(transformed['observation_data']['geo'])

        del batch, transformed

        if i % 50 == 0:
            gc.collect()
            snapshot = tracemalloc.take_snapshot()
            top = snapshot.compare_to(snapshot_before, "lineno")
            total_kb = sum(s.size_diff for s in top) / 1024
            print(f"iter {i:>4}: cumulative Python memory growth = {total_kb:+.1f} KB")

    gc.collect()
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    top = snapshot_after.compare_to(snapshot_before, "lineno")
    total_kb = sum(s.size_diff for s in top) / 1024
    print(f"\nTotal Python memory growth after {N_ITERS} iters: {total_kb:+.1f} KB")
    print("\nTop 5 contributors:")
    for stat in sorted(top, key=lambda s: -s.size_diff)[:5]:
        print(f"  {stat}")


if __name__ == "__main__":
    run_check()
