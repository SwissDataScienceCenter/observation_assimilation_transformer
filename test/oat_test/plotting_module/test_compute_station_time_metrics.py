import types

import torch

from oat.plotting_module.plotter import OATPlotter


def _make_plotter(monkeypatch):
    # 3 stations, 2 timesteps, 1 variable
    obs = torch.tensor(
        [
            [[0.0], [1.0], [2.0]],  # t0
            [[0.0], [1.0], [2.0]],  # t1
        ]
    )  # [T=2, S=3, V=1]
    mask = torch.tensor(
        [
            [[1.0], [1.0], [0.0]],  # station 2 missing at t0
            [[1.0], [1.0], [1.0]],
        ]
    )
    dataset = types.SimpleNamespace(
        station_coords=torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        obs_tensor=obs,
        obs_mask_tensor=mask,
    )
    plotter = OATPlotter(
        predictor=object(),
        dataset=dataset,
        eval_indices=[0, 1, 2],
        n_samples=4,
        device="cpu",
    )

    # deterministic samples exactly equal to the observation -> zero error
    def fake_inference(time_idx, query_coords, predictor=None):
        S = query_coords.shape[0]
        # [S, n_samples, V] all equal to the true obs value per station
        vals = dataset.obs_tensor[time_idx]  # [S, V]
        return vals.unsqueeze(1).repeat(1, 4, 1)

    monkeypatch.setattr(plotter, "_run_inference", fake_inference)
    return plotter


def test_per_timestep_shape_and_nan(monkeypatch):
    plotter = _make_plotter(monkeypatch)
    out = plotter.compute_station_time_metrics(
        time_indices=[0, 1], eval_station_indices=[0, 1, 2]
    )
    assert out["crps"].shape == (2, 3, 1)
    assert out["mae"].shape == (2, 3, 1)
    # station 2 at t0 is masked out -> NaN
    assert torch.isnan(out["crps"][0, 2, 0])
    assert torch.isnan(out["mae"][0, 2, 0])
    # perfect predictions -> zero error where observed
    assert out["mae"][0, 0, 0].item() == 0.0
    assert out["crps"][1, 2, 0].item() == 0.0
