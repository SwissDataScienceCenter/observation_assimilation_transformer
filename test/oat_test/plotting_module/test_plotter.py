import pytest
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unittest.mock import MagicMock

from oat.plotting_module.plotter import OATPlotter


def _make_predictor(N_q: int = 1, n_samples: int = 10, n_vars: int = 2):
    predictor = MagicMock()
    predictor.var_to_indx = {"temp": 0, "wspd": 1}
    predictor.output_var_names = ["temp", "wspd"]
    predictor._normalize_batch.side_effect = lambda batch: batch
    predictor.output_transform.inverse.side_effect = lambda x: x
    predictor.return_value = {"samples": torch.zeros(1, N_q, n_samples, n_vars)}
    return predictor


def _make_dataset(n_stations: int = 5, n_times: int = 4, n_vars: int = 2):
    ds = MagicMock()
    ds.station_coords = torch.zeros(n_stations, 2)
    ds.obs_tensor = torch.zeros(n_times, n_stations, n_vars)
    ds.obs_mask_tensor = torch.ones(n_times, n_stations, n_vars)
    ds.times = pd.date_range("2022-01-01", periods=n_times, freq="h", tz="UTC")
    ds.__getitem__.return_value = {
        "global_data": {
            "global_field": torch.zeros(4, 8, 8),
            "time_emb": torch.zeros(4),
        },
        "observation_data": {
            "coords": torch.zeros(n_stations, 2),
            "geo": torch.zeros(n_stations, 3),
            "topo": torch.zeros(n_stations, 1, 32, 32),
            "obs": torch.zeros(n_stations, n_vars),
            "mask_obs": torch.ones(n_stations, n_vars),
        },
    }
    ds.get_query.return_value = {
        "coords": torch.zeros(1, 2),
        "topo": torch.zeros(1, 1, 32, 32),
        "geo": torch.zeros(1, 3),
    }
    return ds


def test_run_inference_returns_correct_shape():
    N_q, n_samples, n_vars = 3, 10, 2
    predictor = _make_predictor(N_q=N_q, n_samples=n_samples, n_vars=n_vars)
    dataset = _make_dataset()

    plotter = OATPlotter(predictor, dataset, obs_station_indices=[0, 1, 2], n_samples=n_samples)
    query_coords = torch.zeros(N_q, 2)
    samples = plotter._run_inference(time_idx=0, query_coords=query_coords)

    assert samples.shape == (N_q, n_samples, n_vars)


def test_plotter_stores_n_samples():
    predictor = _make_predictor()
    dataset = _make_dataset()
    assert OATPlotter(predictor, dataset, n_samples=42).n_samples == 42


def test_plotter_default_n_samples():
    predictor = _make_predictor()
    dataset = _make_dataset()
    assert OATPlotter(predictor, dataset).n_samples == 100
