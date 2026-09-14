import numpy as np
import pandas as pd
import pytest
import torch

from oat.data_module.data_split import (
    EvalDataset,
    TrainDataset,
    _as_index_array,
    _slice_field,
    create_splits,
)
from oat.data_module.field import Field


def test_as_index_array_integer_list_passthrough():
    result = _as_index_array([1, 3, 2])
    assert np.array_equal(result, np.array([1, 3, 2]))


def test_as_index_array_resolves_station_ids():
    all_ids = np.array(["a", "b", "c", "d"])
    result = _as_index_array(["b", "d"], all_ids)
    assert np.array_equal(result, np.array([1, 3]))


def test_as_index_array_missing_id_raises():
    all_ids = np.array(["a", "b", "c", "d"])
    with pytest.raises(ValueError, match="z"):
        _as_index_array(["b", "z"], all_ids)


def test_as_index_array_string_without_station_ids_raises():
    with pytest.raises(ValueError, match="all_station_ids"):
        _as_index_array(["a", "b"])


class _Base:
    """Minimal base field dataset: 4 stations, 3 times, one 'obs' field."""

    def __init__(self, num_stations=4):
        if num_stations == 4:
            self.stations = np.array(["a", "b", "c", "d"])
        elif num_stations == 6:
            self.stations = np.array(["a", "b", "c", "d", "e", "f"])
        else:
            self.stations = np.array([chr(ord("a") + i) for i in range(num_stations)])
        self.times = pd.date_range("2020-01-01", periods=3, freq="1h", tz="UTC")

    def __len__(self):
        return len(self.times)

    def field_coord_sources(self):
        return {"obs": "stations"}

    def target_coord_source(self):
        return "stations"

    def __getitem__(self, idx):
        coords = torch.tensor(
            [
                [0.0, 0.0, float(idx)],
                [1.0, 0.0, float(idx)],
                [0.0, 1.0, float(idx)],
                [1.0, 1.0, float(idx)],
            ]
        )
        vals = torch.arange(4.0).reshape(4, 1) + idx
        mask = torch.ones(4, 1)
        return {
            "index": idx,
            "fields": {"obs": Field(coords, vals, ["temperature"], mask)},
        }


class _BaseWithGrid(_Base):
    """Mock with one station field and one grid field (2x3=6 grid points)."""

    def field_coord_sources(self):
        return {"obs": "stations", "nwp_grid_field": "nwp_grid"}

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        grid_coords = torch.tensor(
            [[i, j, float(idx)] for i in range(2) for j in range(3)],
            dtype=torch.float32,
        )
        grid_vals = torch.arange(6.0, dtype=torch.float32).reshape(6, 1) + 100 + idx
        sample["fields"]["nwp_grid_field"] = Field(
            grid_coords, grid_vals, ["nwp_t"], None
        )
        return sample


def test_slice_field_copies_and_selects():
    f = Field(
        torch.zeros(4, 3), torch.arange(4.0).reshape(4, 1), ["v"], torch.ones(4, 1)
    )
    s = _slice_field(f, torch.tensor([2, 0]))
    assert s.values.shape == (2, 1)
    assert torch.equal(s.values[:, 0], torch.tensor([2.0, 0.0]))
    assert s.mask is not None and s.mask.shape == (2, 1)


def test_train_dataset_disjoint_context_query():
    base = _Base()
    train = TrainDataset(base, np.array([0, 1, 2, 3]), split_fraction=0.5, seed=0)
    sample = train[0]
    obs = sample["observation_data"]["fields"]["obs"]
    qry = sample["query_data"]["fields"]["obs"]
    assert obs.num_points == 2 and qry.num_points == 2
    # observed + query partition the 4 stations, no overlap
    obs_ids = set(obs.values[:, 0].tolist())
    qry_ids = set(qry.values[:, 0].tolist())
    assert obs_ids.isdisjoint(qry_ids)
    assert obs_ids | qry_ids == {0.0, 1.0, 2.0, 3.0}


def test_eval_dataset_fixed_indices():
    base = _Base()
    ev = EvalDataset(
        base, eval_indices=np.array([3]), observed_indices=np.array([0, 1])
    )
    sample = ev[1]
    assert sample["observation_data"]["fields"]["obs"].num_points == 2
    assert sample["query_data"]["fields"]["obs"].num_points == 1
    # query is station index 3 -> value 3 + time(1) = 4
    assert sample["query_data"]["fields"]["obs"].values[0, 0].item() == 4.0


def test_create_splits_shapes_and_temporal():
    base = _Base()
    out = create_splits(
        base,
        val_station_fraction=0.25,
        test_station_fraction=0.25,
        train_split_fraction=0.5,
        train_station_indices=None,
        val_station_indices=None,
        test_station_indices=None,
        test_days=0,
        seed=1,
    )
    tr, va, te = out["datasets"]
    assert isinstance(tr, TrainDataset)
    assert isinstance(va, EvalDataset) and isinstance(te, EvalDataset)
    assert len(out["station_indices"][0]) == 2  # train stations
    assert len(tr) == 3


def test_train_dataset_grid_field_unsliced():
    """Grid-coord fields bypass station slicing and land in global_data."""
    base = _BaseWithGrid()
    train = TrainDataset(base, np.array([0, 1, 2, 3]), split_fraction=0.5, seed=0)
    sample = train[0]
    assert "global_data" in sample
    grid_f = sample["global_data"]["fields"]["nwp_grid_field"]
    assert grid_f.num_points == 6  # full grid, not sliced to 2
    assert "nwp_grid_field" not in sample["observation_data"]["fields"]
    assert "nwp_grid_field" not in sample["query_data"]["fields"]
    obs_f = sample["observation_data"]["fields"]["obs"]
    assert obs_f.num_points == 2  # station field still sliced


def test_eval_dataset_grid_field_unsliced():
    """Grid-coord fields pass through unsliced in EvalDataset too."""
    base = _BaseWithGrid()
    ev = EvalDataset(
        base, eval_indices=np.array([3]), observed_indices=np.array([0, 1])
    )
    sample = ev[0]
    assert sample["global_data"]["fields"]["nwp_grid_field"].num_points == 6
    assert sample["observation_data"]["fields"]["obs"].num_points == 2
    assert sample["query_data"]["fields"]["obs"].num_points == 1


def test_train_dataset_no_global_data_when_no_grid_fields():
    """When all fields are station-coord, no global_data key appears."""
    base = _Base()
    train = TrainDataset(base, np.array([0, 1, 2, 3]), split_fraction=0.5, seed=0)
    sample = train[0]
    assert "global_data" not in sample


def test_create_splits_with_test_context_stations():
    base = _Base(num_stations=6)
    out = create_splits(
        base,
        train_station_indices=["a", "b"],
        val_station_indices=["c"],
        test_station_indices=["d"],
        test_context_station_indices=["e", "f"],
        test_days=0,
        seed=0,
    )
    tr, va, te = out["datasets"]
    assert set(te.observed_indices.tolist()) == {0, 1, 4, 5}  # a,b,e,f
    assert te.eval_indices.tolist() == [3]  # d only
    assert tr.num_stations == 2
    assert va.observed_indices.tolist() == [0, 1]  # unchanged: just train


def test_create_splits_without_test_context_is_unchanged():
    base = _Base()
    out = create_splits(
        base,
        train_station_indices=["a", "b"],
        val_station_indices=["c"],
        test_station_indices=["d"],
        test_days=0,
        seed=0,
    )
    tr, va, te = out["datasets"]
    assert te.observed_indices.tolist() == [0, 1]  # train+val only, as today
