import numpy as np
import pandas as pd
import torch

from oat.data_module.data_split import (
    EvalDataset,
    TrainDataset,
    _slice_field,
    create_splits,
)
from oat.data_module.field import Field


class _Base:
    """Minimal base field dataset: 4 stations, 3 times, one 'obs' field."""

    def __init__(self):
        self.stations = np.array(["a", "b", "c", "d"])
        self.times = pd.date_range("2020-01-01", periods=3, freq="1h", tz="UTC")

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        coords = torch.tensor(
            [[0.0, 0.0, float(idx)], [1.0, 0.0, float(idx)],
             [0.0, 1.0, float(idx)], [1.0, 1.0, float(idx)]]
        )
        vals = torch.arange(4.0).reshape(4, 1) + idx
        mask = torch.ones(4, 1)
        return {"index": idx, "fields": {"obs": Field(coords, vals, ["temperature"], mask)}}


def test_slice_field_copies_and_selects():
    f = Field(torch.zeros(4, 3), torch.arange(4.0).reshape(4, 1), ["v"], torch.ones(4, 1))
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
    ev = EvalDataset(base, eval_indices=np.array([3]), observed_indices=np.array([0, 1]))
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
