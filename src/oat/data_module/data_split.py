import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, Optional
from oat.data_module import transform, support
from torch.utils.data import DataLoader
from omegaconf import ListConfig


def create_splits(
    dataset: Dataset[Dict[str, Any]],
    logger: logging.Logger | None = None,
    val_station_fraction: float = 0.2,
    test_station_fraction: float = 0.2,
    train_split_fraction: float = 0.8,
    train_station_indices: Optional[np.ndarray | list | ListConfig] = None,
    val_station_indices: Optional[np.ndarray | list | ListConfig] = None,
    test_station_indices: Optional[np.ndarray | list | ListConfig] = None,
    test_days: int = 0,
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Create train, validation, and test splits from a dataset.

    Station split: stations are randomly partitioned into train, val, and test
    roles. The *_station_fraction / *_station_indices parameters control this.

    Temporal split: when test_days > 0, the last `test_days` calendar days are
    withheld as the temporal test period. Train and validation datasets are
    restricted to the earlier time steps; the test dataset uses only the
    held-out time steps.

    Args:
        dataset: Input dataset with .stations (array) and .times (DatetimeIndex).
        logger: Optional logger.
        val_station_fraction: Fraction of stations reserved for validation.
        test_station_fraction: Fraction of stations reserved for testing.
        train_split_fraction: Fraction of training stations used as observed
            (vs query) stations within each training sample.
        train_station_indices: Optional explicit train station indices.
        val_station_indices: Optional explicit validation station indices.
        test_station_indices: Optional explicit test station indices.
        test_days: Number of days from the end of the dataset to hold out as
            temporal test. 0 means no temporal split.
        seed: Random seed for reproducibility.

    Returns:
        Dict with:
          "datasets": (train_dataset, val_dataset, test_dataset)
          "station_indices": (train_list, val_list, test_list)
          "test_time_indices": list of dataset-level time positions in the test
              period (empty list when test_days=0).
    """
    # ── Temporal split ────────────────────────────────────────────────────────
    all_positions = np.arange(len(dataset))
    if test_days > 0 and not hasattr(dataset, "times"):
        raise AttributeError("test_days > 0 requires dataset.times (pandas DatetimeIndex).")
    if test_days > 0:
        cutoff = dataset.times[-1] - pd.Timedelta(days=test_days)
        train_val_mask = dataset.times <= cutoff
        train_val_time_indices: np.ndarray | None = all_positions[train_val_mask]
        test_time_indices: np.ndarray = all_positions[~train_val_mask]
        if logger:
            logger.info(
                f"Temporal split: {len(train_val_time_indices)} train/val time steps, "
                f"{len(test_time_indices)} test time steps (last {test_days} days)."
            )
    else:
        train_val_time_indices = None
        test_time_indices = np.array([], dtype=int)

    # ── Station split ─────────────────────────────────────────────────────────
    num_stations = len(dataset.stations)
    test_size = int(num_stations * test_station_fraction)
    val_size = int(num_stations * val_station_fraction)

    if all(x is None for x in (train_station_indices, val_station_indices, test_station_indices)):
        assert val_station_fraction + test_station_fraction < 1.0, (
            "val_station_fraction and test_station_fraction must sum to less than 1."
        )
        assert val_station_fraction >= 0.0 and test_station_fraction >= 0.0, (
            "val_station_fraction and test_station_fraction must be non-negative."
        )

        indices = np.arange(num_stations, dtype=int)
        rng = torch.Generator().manual_seed(seed) if seed is not None else None
        indices = indices[torch.randperm(num_stations, generator=rng).numpy()]

        train_station_indices = indices[: num_stations - (val_size + test_size)]
        val_station_indices = indices[num_stations - (val_size + test_size) : num_stations - test_size]
        test_station_indices = indices[num_stations - test_size :]

        if logger:
            logger.info("Generated new station split indices.")
            logger.info(f"Train station indices: {train_station_indices}")
            logger.info(f"Val station indices: {val_station_indices}")
            logger.info(f"Test station indices: {test_station_indices}")
    else:
        assert all(x is not None for x in (train_station_indices, val_station_indices, test_station_indices))
        if logger:
            logger.info("Using station split indices from configuration.")
            logger.info(f"Train station indices: {train_station_indices}")
            logger.info(f"Val station indices: {val_station_indices}")
            logger.info(f"Test station indices: {test_station_indices}")

        train_station_indices = (
            np.array(train_station_indices)
            if isinstance(train_station_indices, (list, ListConfig))
            else train_station_indices
        )
        val_station_indices = (
            np.array(val_station_indices)
            if isinstance(val_station_indices, (list, ListConfig))
            else val_station_indices
        )
        test_station_indices = (
            np.array(test_station_indices)
            if isinstance(test_station_indices, (list, ListConfig))
            else test_station_indices
        )

        if logger:
            if len(val_station_indices) != val_size:
                logger.warning(
                    f"Length of val_station_indices ({len(val_station_indices)}) "
                    f"does not match val_station_fraction {val_station_fraction}. "
                    "Make sure this is intentional!"
                )
            if len(test_station_indices) != test_size:
                logger.warning(
                    f"Length of test_station_indices ({len(test_station_indices)}) "
                    f"does not match test_station_fraction {test_station_fraction}. "
                    "Make sure this is intentional!"
                )
            if len(train_station_indices) != (num_stations - val_size - test_size):
                logger.warning(
                    f"Length of train_station_indices ({len(train_station_indices)}) "
                    f"does not match fractions {val_station_fraction}/{test_station_fraction}. "
                    "Make sure this is intentional!"
                )

    assert (
        len(train_station_indices) > 0
        and len(val_station_indices) > 0
        and len(test_station_indices) > 0
    ), "Each station split must contain at least one station."

    # ── Assemble datasets ─────────────────────────────────────────────────────
    train_dataset = TrainDataset(
        dataset, train_station_indices,
        split_fraction=train_split_fraction,
        seed=seed,
        time_indices=train_val_time_indices,
    )

    observed_for_val = (
        np.array([], dtype=int) if train_split_fraction == 0.0 else train_station_indices
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
            train_station_indices.tolist(),
            val_station_indices.tolist(),
            test_station_indices.tolist(),
        ),
        "test_time_indices": test_time_indices.tolist(),
    }


class TrainDataset(Dataset[Dict[str, Any]]):
    """
    A dataset wrapper that splits training station data into observed and query sets.
    Randomly splits train stations into observed (keys) and query for each sample.
    Optionally restricted to a subset of time steps via time_indices.

    Args:
        dataset: The underlying dataset.
        train_indices: Array of station indices for training.
        split_fraction: Fraction of stations used as observed/key stations (default 0.8).
        seed: Optional random seed.
        time_indices: Optional array of dataset-level time positions to use.
            When None all time steps are used.
    """

    def __init__(
        self,
        dataset: Dataset[Dict[str, Any]],
        train_indices: np.ndarray,
        split_fraction: float = 0.8,
        seed: int | None = None,
        time_indices: np.ndarray | None = None,
    ):
        self.dataset = dataset
        self.train_indices = train_indices
        self.stations = dataset.stations[self.train_indices]
        self.split_fraction = split_fraction
        self.rng = torch.Generator().manual_seed(seed) if seed is not None else None
        self._time_indices = time_indices

    def __len__(self) -> int:
        if self._time_indices is not None:
            return len(self._time_indices)
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        actual_idx = self._time_indices[idx] if self._time_indices is not None else idx
        data = self.dataset[actual_idx]

        permuted = self.train_indices[
            torch.randperm(len(self.train_indices), generator=self.rng).numpy()
        ]
        split_point = self.num_observed_stations
        key_indices = permuted[:split_point]
        query_indices = permuted[split_point:]

        observation_data = data["observation_data"]
        key_data, query_data = {}, {}
        for key in observation_data:
            key_data[key] = observation_data[key][key_indices]
            query_data[key] = observation_data[key][query_indices]

        key_data["station_index"] = key_indices
        query_data["station_index"] = query_indices

        return data | {"observation_data": key_data, "query_data": query_data}

    @property
    def obs_var_support(self) -> support.VariableSupport:
        return self.dataset.obs_var_support

    @property
    def station_names(self) -> list[str]:
        return self.stations.tolist()

    @property
    def station_info(self) -> pd.DataFrame:
        return self.dataset.obs_ds.stations_table.loc[self.stations]

    @property
    def variable_names(self) -> dict[str, Any]:
        return self.dataset.variable_names

    @property
    def num_stations(self) -> int:
        return len(self.train_indices)

    @property
    def num_observed_stations(self) -> int:
        return int(len(self.train_indices) * self.split_fraction)

    @property
    def num_query_stations(self) -> int:
        return len(self.train_indices) - self.num_observed_stations

    def get_transform(
        self,
        num_data_points: int,
        dataloader: DataLoader | None = None,
        type: str = "real",
    ) -> tuple[transform.NestedTransform, transform.Transform]:
        """Compute data-driven transforms from random samples.

        Returns: (NestedTransform for batches, Transform for obs-only output).
        """
        data_indices = None if dataloader is not None else self._time_indices
        return self.dataset.get_transform(
            num_data_points=num_data_points,
            dataloader=dataloader,
            type=type,
            data_indices=data_indices,
        )


class EvalDataset(Dataset[Dict[str, Any]]):
    """
    A dataset wrapper that splits stations into evaluation (query) and observed sets.
    Optionally restricted to a subset of time steps via time_indices.

    Args:
        dataset: The underlying dataset with a 'stations' attribute.
        eval_indices: Station indices used as query / evaluation targets.
        observed_indices: Station indices used as context observations.
        time_indices: Optional array of dataset-level time positions to use.
            When None all time steps are used.
    """

    def __init__(
        self,
        dataset: Dataset[Dict[str, Any]],
        eval_indices: np.ndarray,
        observed_indices: np.ndarray,
        time_indices: np.ndarray | None = None,
    ):
        assert hasattr(dataset, "stations"), "Dataset must have a stations attribute."
        assert len(dataset.stations) > 0, "Dataset stations must not be empty."
        self.dataset = dataset
        self.eval_indices = eval_indices
        self.observed_indices = observed_indices
        self._time_indices = time_indices
        self.eval_stations = dataset.stations[self.eval_indices]
        self.observed_stations = dataset.stations[self.observed_indices]

    def __len__(self) -> int:
        if self._time_indices is not None:
            return len(self._time_indices)
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        actual_idx = self._time_indices[idx] if self._time_indices is not None else idx
        data = self.dataset[actual_idx]
        observation_data = data["observation_data"]
        eval_data, observed_data = {}, {}
        for key in observation_data:
            eval_data[key] = observation_data[key][self.eval_indices]
            observed_data[key] = observation_data[key][self.observed_indices]
        return data | {"observation_data": observed_data, "query_data": eval_data}

    @property
    def station_names(self) -> dict[str, list[str]]:
        return {"eval": self.eval_stations.tolist(), "observed": self.observed_stations.tolist()}

    @property
    def station_info(self) -> dict[str, pd.DataFrame]:
        return {
            "eval": self.dataset.obs_ds.stations_table.loc[self.eval_stations],
            "observed": self.dataset.obs_ds.stations_table.loc[self.observed_stations],
        }

    @property
    def variable_names(self) -> dict[str, Any]:
        return self.dataset.variable_names

    @property
    def num_stations(self) -> int:
        return len(self.eval_indices) + len(self.observed_indices)

    @property
    def num_observed_stations(self) -> int:
        return len(self.observed_indices)

    @property
    def num_query_stations(self) -> int:
        return len(self.eval_indices)
