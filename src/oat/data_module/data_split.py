"""Field-native train/eval split wrappers mirroring legacy/data_split.py semantics."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import ListConfig
from torch import Tensor
from torch.utils.data import Dataset

from oat.data_module.field import Field


def _slice_field(field: Field, idx: Tensor) -> Field:
    """Point-slice a Field along its points dim with an index tensor (copy)."""
    mask = field.mask[idx] if field.mask is not None else None
    return Field(field.coords[idx], field.values[idx], list(field.var_names), mask)


def _as_index_array(x: Any) -> np.ndarray:  # type: ignore[type-arg]
    if isinstance(x, (list, ListConfig)):
        return np.array(x)
    return x  # type: ignore[no-any-return]


class TrainDataset(Dataset[dict[str, Any]]):
    """Random per-sample observed/query split over a set of train stations."""

    def __init__(
        self,
        dataset: Any,
        train_indices: np.ndarray,  # type: ignore[type-arg]
        split_fraction: float = 0.8,
        seed: int | None = None,
        time_indices: np.ndarray | None = None,  # type: ignore[type-arg]
    ) -> None:
        self.dataset = dataset
        self.train_indices = np.asarray(train_indices)
        self.stations = dataset.stations[self.train_indices]
        self.split_fraction = split_fraction
        self.rng = torch.Generator().manual_seed(seed) if seed is not None else None
        self._time_indices = time_indices

    def __len__(self) -> int:
        if self._time_indices is not None:
            return len(self._time_indices)
        return len(self.dataset)

    @property
    def num_stations(self) -> int:
        return len(self.train_indices)

    @property
    def num_observed_stations(self) -> int:
        return int(len(self.train_indices) * self.split_fraction)

    @property
    def num_query_stations(self) -> int:
        return len(self.train_indices) - self.num_observed_stations

    def __getitem__(self, idx: int) -> dict[str, Any]:
        actual_idx = (
            int(self._time_indices[idx]) if self._time_indices is not None else idx
        )
        fields = self.dataset[actual_idx]["fields"]
        permuted = self.train_indices[
            torch.randperm(len(self.train_indices), generator=self.rng).numpy()
        ]
        split_point = self.num_observed_stations
        key_idx = torch.as_tensor(permuted[:split_point], dtype=torch.long)
        query_idx = torch.as_tensor(permuted[split_point:], dtype=torch.long)
        obs_fields = {n: _slice_field(f, key_idx) for n, f in fields.items()}
        query_fields = {n: _slice_field(f, query_idx) for n, f in fields.items()}
        return {
            "index": actual_idx,
            "observation_data": {"fields": obs_fields},
            "query_data": {"fields": query_fields},
        }

    def get_transform(self, num_data_points: int) -> Any:
        # Field path assembles the two-group transform in setup.py; this
        # delegates stats gathering to the base dataset over train stations.
        return self.dataset.get_transform(
            num_data_points=num_data_points,
            data_indices=self._time_indices,
            target_point_indices=self.train_indices,
        )


class EvalDataset(Dataset[dict[str, Any]]):
    """Fixed observed/query station split for validation / test."""

    def __init__(
        self,
        dataset: Any,
        eval_indices: np.ndarray,  # type: ignore[type-arg]
        observed_indices: np.ndarray,  # type: ignore[type-arg]
        time_indices: np.ndarray | None = None,  # type: ignore[type-arg]
    ) -> None:
        self.dataset = dataset
        self.eval_indices = torch.as_tensor(np.asarray(eval_indices), dtype=torch.long)
        self.observed_indices = torch.as_tensor(
            np.asarray(observed_indices), dtype=torch.long
        )
        self._time_indices = time_indices
        self.stations = dataset.stations

    def __len__(self) -> int:
        if self._time_indices is not None:
            return len(self._time_indices)
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        actual_idx = (
            int(self._time_indices[idx]) if self._time_indices is not None else idx
        )
        fields = self.dataset[actual_idx]["fields"]
        obs_fields = {
            n: _slice_field(f, self.observed_indices) for n, f in fields.items()
        }
        query_fields = {
            n: _slice_field(f, self.eval_indices) for n, f in fields.items()
        }
        return {
            "index": actual_idx,
            "observation_data": {"fields": obs_fields},
            "query_data": {"fields": query_fields},
        }


def create_splits(
    dataset: Any,
    *,
    val_station_fraction: float = 0.2,
    test_station_fraction: float = 0.2,
    train_split_fraction: float = 0.8,
    train_station_indices: Any = None,
    val_station_indices: Any = None,
    test_station_indices: Any = None,
    test_days: int = 0,
    seed: int | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Field-native mirror of legacy.data_split.create_splits (same semantics)."""
    all_positions = np.arange(len(dataset))
    if test_days > 0:
        cutoff = dataset.times[-1] - pd.Timedelta(days=test_days)
        train_val_mask = dataset.times <= cutoff
        train_val_time_indices: np.ndarray | None = all_positions[train_val_mask]  # type: ignore[type-arg]
        test_time_indices = all_positions[~train_val_mask]
    else:
        train_val_time_indices = None
        test_time_indices = np.array([], dtype=int)

    num_stations = len(dataset.stations)
    test_size = int(num_stations * test_station_fraction)
    val_size = int(num_stations * val_station_fraction)

    if all(
        x is None
        for x in (train_station_indices, val_station_indices, test_station_indices)
    ):
        assert val_station_fraction + test_station_fraction < 1.0
        indices = np.arange(num_stations, dtype=int)
        rng = torch.Generator().manual_seed(seed) if seed is not None else None
        indices = indices[torch.randperm(num_stations, generator=rng).numpy()]
        train_station_indices = indices[: num_stations - (val_size + test_size)]
        val_station_indices = indices[
            num_stations - (val_size + test_size) : num_stations - test_size
        ]
        test_station_indices = indices[num_stations - test_size :]
    else:
        train_station_indices = _as_index_array(train_station_indices)
        val_station_indices = _as_index_array(val_station_indices)
        test_station_indices = _as_index_array(test_station_indices)

    assert (
        len(train_station_indices) > 0
        and len(val_station_indices) > 0
        and len(test_station_indices) > 0
    )

    train_dataset = TrainDataset(
        dataset,
        train_station_indices,
        split_fraction=train_split_fraction,
        seed=seed,
        time_indices=train_val_time_indices,
    )
    observed_for_val = (
        np.array([], dtype=int)
        if train_split_fraction == 0.0
        else train_station_indices
    )
    val_dataset = EvalDataset(
        dataset,
        eval_indices=val_station_indices,
        observed_indices=observed_for_val,
        time_indices=train_val_time_indices,
    )
    observed_for_test = (
        np.array([], dtype=int)
        if train_split_fraction == 0.0
        else np.concatenate((train_station_indices, val_station_indices))
    )
    test_time_indices_arg = test_time_indices if test_days > 0 else None
    test_dataset = EvalDataset(
        dataset,
        eval_indices=test_station_indices,
        observed_indices=observed_for_test,
        time_indices=test_time_indices_arg,
    )
    return {
        "datasets": (train_dataset, val_dataset, test_dataset),
        "station_indices": (
            np.asarray(train_station_indices).tolist(),
            np.asarray(val_station_indices).tolist(),
            np.asarray(test_station_indices).tolist(),
        ),
        "test_time_indices": test_time_indices.tolist(),
    }
