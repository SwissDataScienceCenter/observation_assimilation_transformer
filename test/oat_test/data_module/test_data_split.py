import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset
from typing import Any, Dict

from oat.data_module.data_split import create_splits, TrainDataset, EvalDataset


class MockDataset(Dataset):
    """Minimal dataset for unit-testing data_split logic without real data."""

    def __init__(self, num_stations: int = 20, num_times: int = 100):
        self.stations = np.array([f"S{i:03d}" for i in range(num_stations)])
        self.times = pd.date_range("2021-01-01", periods=num_times, freq="h", tz="UTC")
        self._num_times = num_times
        self._num_stations = num_stations

    def __len__(self) -> int:
        return self._num_times

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            # Encode idx in global_field so callers can verify which time step was fetched.
            "global_data": {"global_field": torch.full((3, 4, 4), float(idx))},
            "observation_data": {
                "obs": torch.zeros(self._num_stations, 2),
                "coords": torch.zeros(self._num_stations, 2),
                "geo": torch.zeros(self._num_stations, 3),
                "topo": torch.zeros(self._num_stations, 1, 8, 8),
                "mask_obs": torch.zeros(self._num_stations, 2, dtype=torch.bool),
            },
        }


# ── TrainDataset with time_indices ───────────────────────────────────────────

def test_train_dataset_no_time_indices_uses_all_times():
    ds = MockDataset(num_times=100, num_stations=20)
    train_ds = TrainDataset(ds, np.arange(12))
    assert len(train_ds) == 100


def test_train_dataset_time_indices_restricts_length():
    ds = MockDataset(num_times=100, num_stations=20)
    train_ds = TrainDataset(ds, np.arange(12), time_indices=np.arange(80))
    assert len(train_ds) == 80


def test_train_dataset_getitem_maps_through_time_indices():
    ds = MockDataset(num_times=100, num_stations=20)
    time_idx = np.array([10, 20, 30])
    # split_fraction=1.0 so all stations are "observed" — keeps __getitem__ simple
    train_ds = TrainDataset(ds, np.arange(3), time_indices=time_idx, split_fraction=1.0)
    # local idx 0 → ds[time_idx[0]] = ds[10], global_field filled with 10.0
    item = train_ds[0]
    assert item["global_data"]["global_field"][0, 0, 0].item() == pytest.approx(10.0)
    item = train_ds[2]
    assert item["global_data"]["global_field"][0, 0, 0].item() == pytest.approx(30.0)


# ── EvalDataset with time_indices ────────────────────────────────────────────

def test_eval_dataset_no_time_indices_uses_all_times():
    ds = MockDataset(num_times=100, num_stations=20)
    eval_ds = EvalDataset(ds, eval_indices=np.arange(5), observed_indices=np.arange(5, 15))
    assert len(eval_ds) == 100


def test_eval_dataset_time_indices_restricts_length():
    ds = MockDataset(num_times=100, num_stations=20)
    time_idx = np.arange(80, 100)  # 20 test time steps
    eval_ds = EvalDataset(ds, eval_indices=np.arange(5), observed_indices=np.arange(5, 15),
                          time_indices=time_idx)
    assert len(eval_ds) == 20


def test_eval_dataset_getitem_maps_through_time_indices():
    ds = MockDataset(num_times=100, num_stations=20)
    time_idx = np.array([50, 60, 70])
    eval_ds = EvalDataset(ds, eval_indices=np.arange(5), observed_indices=np.arange(5, 15),
                          time_indices=time_idx)
    item = eval_ds[1]  # local idx 1 → ds[60]
    assert item["global_data"]["global_field"][0, 0, 0].item() == pytest.approx(60.0)


# ── create_splits — station rename ───────────────────────────────────────────

def test_station_fraction_params_respected():
    """val/test_station_fraction correctly size station groups."""
    ds = MockDataset(num_stations=20, num_times=50)
    result = create_splits(ds, val_station_fraction=0.2, test_station_fraction=0.2, seed=0)
    train_ds, val_ds, test_ds = result["datasets"]
    # 20 stations → 12 train, 4 val, 4 test
    assert train_ds.num_stations == 12
    assert val_ds.num_query_stations == 4
    assert test_ds.num_query_stations == 4


def test_station_indices_params_override_fractions():
    """Explicit station indices are used instead of computing from fractions."""
    ds = MockDataset(num_stations=20, num_times=50)
    result = create_splits(
        ds,
        train_station_indices=list(range(10)),
        val_station_indices=list(range(10, 15)),
        test_station_indices=list(range(15, 20)),
    )
    _, _, test_ds = result["datasets"]
    assert test_ds.num_query_stations == 5


def test_station_indices_returned_in_result():
    ds = MockDataset(num_stations=20, num_times=50)
    result = create_splits(ds, val_station_fraction=0.2, test_station_fraction=0.2, seed=0)
    train_idx, val_idx, test_idx = result["station_indices"]
    assert len(train_idx) == 12
    assert len(val_idx) == 4
    assert len(test_idx) == 4
    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx).isdisjoint(test_idx)
    assert set(val_idx).isdisjoint(test_idx)


# ── create_splits — temporal test split ─────────────────────────────────────

def test_no_temporal_split_uses_all_times():
    """test_days=0 (default): all three datasets span all time steps."""
    ds = MockDataset(num_stations=20, num_times=100)
    result = create_splits(ds, val_station_fraction=0.2, test_station_fraction=0.2, seed=0)
    train_ds, val_ds, test_ds = result["datasets"]
    assert len(train_ds) == 100
    assert len(val_ds) == 100
    assert len(test_ds) == 100


def test_temporal_split_excludes_test_days_from_train_val():
    """
    200 hourly steps; last 5 days = 120 hours.
    train/val get 80 steps, test gets 120.
    """
    ds = MockDataset(num_stations=20, num_times=200)
    result = create_splits(ds, val_station_fraction=0.2, test_station_fraction=0.2,
                           test_days=5, seed=0)
    train_ds, val_ds, test_ds = result["datasets"]
    assert len(train_ds) == 80
    assert len(val_ds) == 80
    assert len(test_ds) == 120


def test_temporal_split_cutoff_is_last_n_days():
    """All test time steps must be strictly after (last_time - test_days)."""
    ds = MockDataset(num_stations=20, num_times=200)
    result = create_splits(ds, val_station_fraction=0.2, test_station_fraction=0.2,
                           test_days=5, seed=0)
    test_time_indices = result["test_time_indices"]
    expected_cutoff = ds.times[-1] - pd.Timedelta(days=5)
    assert (ds.times[test_time_indices] > expected_cutoff).all()
    non_test = sorted(set(range(len(ds))) - set(test_time_indices))
    assert (ds.times[non_test] <= expected_cutoff).all()


def test_temporal_split_zero_test_days_empty_test_time_indices() -> None:
    ds = MockDataset(num_stations=20, num_times=100)
    result = create_splits(ds, val_station_fraction=0.2, test_station_fraction=0.2,
                           test_days=0, seed=0)
    assert result["test_time_indices"] == []


def test_train_val_and_test_time_indices_partition_all_times():
    """train/val and test time indices cover all time steps exactly once."""
    ds = MockDataset(num_stations=20, num_times=200)
    result = create_splits(ds, val_station_fraction=0.2, test_station_fraction=0.2,
                           test_days=5, seed=0)
    test_idx = set(result["test_time_indices"])
    train_val_size = len(result["datasets"][0])
    train_val_idx = set(range(len(ds))) - test_idx
    assert len(train_val_idx) == train_val_size
    assert train_val_idx.isdisjoint(test_idx)
    assert train_val_idx | test_idx == set(range(len(ds)))
