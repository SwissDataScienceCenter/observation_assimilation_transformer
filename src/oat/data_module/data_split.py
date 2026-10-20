"""Field-native train/eval split wrappers."""

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


def _as_timestamp(value: str, times: pd.DatetimeIndex) -> pd.Timestamp:
    """Parse ``value`` into the timezone of ``times`` (naive means that zone)."""
    ts = pd.Timestamp(value)
    if ts.tz is None and times.tz is not None:
        ts = ts.tz_localize(times.tz)
    return ts


def _check_covers_test_window(
    test_times: pd.DatetimeIndex,
    times: pd.DatetimeIndex,
    test_start_date: str,
    test_end_date: str | None,
) -> None:
    """Raise if ``times`` misses a step of the fixed test window.

    A source that ends early or drops dates (e.g. COSMO-E1 missing hours)
    would otherwise score a shorter window than the other models, and
    the paired comparison would mix different periods without a warning.
    """
    start = _as_timestamp(test_start_date, times)
    end = times[-1] if test_end_date is None else _as_timestamp(test_end_date, times)
    step = times.to_series().diff().min()
    expected = pd.date_range(start, end, freq=step)
    missing = expected.difference(test_times)
    if len(missing) > 0:
        raise ValueError(
            f"Data source does not cover the test window [{start}, {end}]: "
            f"{len(missing)} of {len(expected)} steps are missing "
            f"(first {missing[0]}, last {missing[-1]}). The source covers "
            f"[{times[0]}, {times[-1]}]."
        )


def _as_index_array(
    x: Any, all_station_ids: np.ndarray | None = None  # type: ignore[type-arg]
) -> np.ndarray:  # type: ignore[type-arg]
    if isinstance(x, (list, ListConfig)):
        x = np.array(x)
    if x.dtype.kind in "iu":
        # Check the bounds only when we have a station list to check against.
        # The random-split branch calls this with all_station_ids=None, so
        # there is nothing to bound against there.
        if all_station_ids is not None and x.size:
            if x.min() < 0 or x.max() >= len(all_station_ids):
                raise ValueError(
                    f"station index out of range: values must be in "
                    f"[0, {len(all_station_ids)}), got min={x.min()}, "
                    f"max={x.max()}"
                )
        return x
    if all_station_ids is None:
        raise ValueError("all_station_ids is required to resolve station-ID overrides")
    id_to_pos = {sid: i for i, sid in enumerate(all_station_ids)}
    missing = [sid for sid in x if sid not in id_to_pos]
    if missing:
        raise ValueError(f"station IDs not found in dataset.stations: {missing}")
    return np.array([id_to_pos[sid] for sid in x])


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
        all_fields = self.dataset[actual_idx]["fields"]

        target_cs = self.dataset.target_coord_source()
        fcs = self.dataset.field_coord_sources()
        station_fields = {n: f for n, f in all_fields.items() if fcs[n] == target_cs}
        grid_fields = {n: f for n, f in all_fields.items() if fcs[n] != target_cs}

        permuted = self.train_indices[
            torch.randperm(len(self.train_indices), generator=self.rng).numpy()
        ]
        split_point = self.num_observed_stations
        key_idx = torch.as_tensor(permuted[:split_point], dtype=torch.long)
        query_idx = torch.as_tensor(permuted[split_point:], dtype=torch.long)
        obs_fields = {n: _slice_field(f, key_idx) for n, f in station_fields.items()}
        query_fields = {
            n: _slice_field(f, query_idx) for n, f in station_fields.items()
        }
        out: dict[str, Any] = {
            "index": actual_idx,
            "observation_data": {"fields": obs_fields},
            "query_data": {"fields": query_fields},
        }
        if grid_fields:
            out["global_data"] = {"fields": grid_fields}
        return out

    def get_transform(self, num_data_points: int, type: str = "real") -> Any:
        # Field path assembles the two-group transform in setup.py; this
        # delegates stats gathering to the base dataset over train stations.
        return self.dataset.get_transform(
            num_data_points=num_data_points,
            data_indices=self._time_indices,
            target_point_indices=self.train_indices,
            type=type,
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
        all_fields = self.dataset[actual_idx]["fields"]

        target_cs = self.dataset.target_coord_source()
        fcs = self.dataset.field_coord_sources()
        station_fields = {n: f for n, f in all_fields.items() if fcs[n] == target_cs}
        grid_fields = {n: f for n, f in all_fields.items() if fcs[n] != target_cs}

        obs_fields = {
            n: _slice_field(f, self.observed_indices) for n, f in station_fields.items()
        }
        query_fields = {
            n: _slice_field(f, self.eval_indices) for n, f in station_fields.items()
        }
        out: dict[str, Any] = {
            "index": actual_idx,
            "observation_data": {"fields": obs_fields},
            "query_data": {"fields": query_fields},
        }
        if grid_fields:
            out["global_data"] = {"fields": grid_fields}
        return out


def create_splits(
    dataset: Any,
    *,
    val_station_fraction: float = 0.2,
    test_station_fraction: float = 0.2,
    train_split_fraction: float = 0.8,
    train_station_indices: Any = None,
    val_station_indices: Any = None,
    test_station_indices: Any = None,
    test_context_station_indices: Any = None,
    test_days: int = 0,
    test_start_date: str | None = None,
    test_end_date: str | None = None,
    seed: int | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Create field-native splits.

    ``test_start_date``/``test_end_date`` (inclusive) pin the test window to
    fixed dates and override ``test_days``. Train/val use only the times
    before ``test_start_date``. Use them to compare NWP sources with
    different archive end dates on the same test window: ``test_days``
    counts back from the last time of each archive.
    """
    all_positions = np.arange(len(dataset))
    if test_end_date is not None and test_start_date is None:
        raise ValueError("test_end_date needs test_start_date")
    temporal_split = test_start_date is not None or test_days > 0
    if test_start_date is not None:
        times = dataset.times
        train_val_mask = times < _as_timestamp(test_start_date, times)
        test_mask = ~train_val_mask
        if test_end_date is not None:
            test_mask &= times <= _as_timestamp(test_end_date, times)
        train_val_time_indices: np.ndarray | None = all_positions[train_val_mask]  # type: ignore[type-arg]
        test_time_indices = all_positions[test_mask]
        _check_covers_test_window(
            times[test_mask], times, test_start_date, test_end_date
        )
    elif test_days > 0:
        cutoff = dataset.times[-1] - pd.Timedelta(days=test_days)
        train_val_mask = dataset.times <= cutoff
        train_val_time_indices = all_positions[train_val_mask]
        test_time_indices = all_positions[~train_val_mask]
    else:
        train_val_time_indices = None
        test_time_indices = np.array([], dtype=int)

    num_stations = len(dataset.stations)
    test_size = int(num_stations * test_station_fraction)
    val_size = int(num_stations * val_station_fraction)

    resolved_from_override = not all(
        x is None
        for x in (train_station_indices, val_station_indices, test_station_indices)
    )
    if not resolved_from_override:
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
        train_station_indices = _as_index_array(train_station_indices, dataset.stations)
        val_station_indices = _as_index_array(val_station_indices, dataset.stations)
        test_station_indices = _as_index_array(test_station_indices, dataset.stations)
    test_context_station_indices = (
        np.array([], dtype=int)
        if test_context_station_indices is None
        else _as_index_array(test_context_station_indices, dataset.stations)
    )

    if resolved_from_override:
        # Check that the four explicit station groups do not share a
        # station. The random-split branch is disjoint by construction and
        # does not need this check.
        groups = (
            train_station_indices,
            val_station_indices,
            test_station_indices,
            test_context_station_indices,
        )
        all_indices = np.concatenate(groups)
        if len(all_indices) != len(set(all_indices.tolist())):
            raise ValueError(
                "station index groups (train/val/test/test_context) must "
                "be disjoint; found overlapping station indices"
            )

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
        else np.concatenate(
            (train_station_indices, val_station_indices, test_context_station_indices)
        )
    )
    test_time_indices_arg = test_time_indices if temporal_split else None
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
